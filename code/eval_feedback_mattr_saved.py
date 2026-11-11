"""Evaluate saved feedback MAttr scores; no optimizer, backward, or training call.
Reuses only the preparation portion of the user's adapter and its intervention API.
"""
import argparse
import ast
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import subprocess

import torch

LOG = logging.getLogger("feedback_mattr_eval")
FRACTIONS = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def prepare_function(adapter, adapter_path):
    tree = ast.parse(adapter_path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    starts = [i for i,n in enumerate(main.body)
              if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and ast.unparse(n.value.func) == "torch.set_num_threads"]
    stops = [i for i,n in enumerate(main.body) if assignment(n, "train_env")]
    if len(starts) != 1 or len(stops) != 1 or starts[0] >= stops[0]:
        raise RuntimeError("Adapter structure changed: cannot safely isolate preparation.")
    body = copy.deepcopy(main.body[starts[0]:stops[0]])
    body = [n for n in body if not (isinstance(n, ast.ImportFrom)
            and n.module == "matryoshka_attribution" and any(a.name == "learn_scores" for a in n.names))]
    changed = 0
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "mkdtemp":
            for kw in node.keywords:
                if kw.arg == "prefix" and isinstance(kw.value, ast.Constant) and kw.value.value == "feedback_mattr_":
                    kw.value.value = "feedback_mattr_eval_"
                    changed += 1
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name == "learn_scores" or name.endswith(".backward") or name.endswith(".step"):
                raise RuntimeError("Training operation found in isolated preparation; stopping.")
    if changed != 1:
        raise RuntimeError("Unexpected output-directory creation in adapter.")
    function = ast.parse("def prepare(args):\n    pass\n").body[0]
    function.body = body + ast.parse("return model, graph, evaluate_examples, factory, out, precision_baseline").body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(adapter))
    exec(compile(module, str(adapter_path) + ":eval_preparation_only", "exec"), namespace)
    return namespace["prepare"], ast.unparse(module)


def average(values):
    return sum(values) / len(values) if values else None


def metrics(current, refs):
    if not all(math.isfinite(m) for m in current):
        raise RuntimeError("Nonfinite circuit margin")
    faith = [(m-r["source"])/(r["base"]-r["source"])
             for m,r in zip(current, refs) if abs(r["base"]-r["source"]) > 1e-6]
    agreement = [(m > 0) == (r["base"] > 0)
                 for m,r in zip(current, refs) if r["base"] != 0]
    return {"IIA": average([m > 0 for m in current]),
            "full_model_base_IIA": average([r["base"] > 0 for r in refs]),
            "agreement_with_full_non_ties": average(agreement),
            "faithfulness_nonzero_denominators": average(faith),
            "faithfulness_defined_examples": len(faith),
            "mean_circuit_margin": average(current),
            "mean_absolute_margin_error_vs_full": average([abs(m-r["base"]) for m,r in zip(current, refs)]),
            "zero_circuit_margins": sum(m == 0 for m in current),
            "evaluation_pairs": len(refs)}


def compactness(curve):
    return sum(math.log(b["fraction"]/a["fraction"]) * (a["IIA"]+b["IIA"])/2
               for a,b in zip(curve,curve[1:])) / math.log(curve[-1]["fraction"]/curve[0]["fraction"])


def export_names(graph, scores, out):
    coords = graph.real_edge_mask.bool().nonzero()
    top = torch.argsort(scores, descending=True)[:20].tolist()
    wanted = {tuple(int(x) for x in coords[i]) for i in top}
    dimensions = tuple(graph.scores.shape)
    resolved, negatives = {}, 0
    for edge in graph.edges.values():
        raw = tuple(int(x) for x in edge.matrix_index)
        if not all(-size <= value < size for value,size in zip(raw, dimensions)):
            raise RuntimeError(f"Out-of-bounds edge coordinate: {raw}")
        negatives += int(any(value < 0 for value in raw))
        canonical = tuple(value % size for value,size in zip(raw, dimensions))
        if canonical in wanted:
            if canonical in resolved and resolved[canonical][0] != edge.name:
                raise RuntimeError(f"Conflicting edge names at {canonical}")
            resolved[canonical] = (edge.name, raw)
    rows, missing = [], []
    for rank,index in enumerate(top,1):
        coordinate = tuple(int(x) for x in coords[index])
        found = resolved.get(coordinate)
        if found is None:
            missing.append(list(coordinate))
        rows.append({"rank":rank,"score_index":index,"matrix_row":coordinate[0],
                     "matrix_column":coordinate[1],"edge":found[0] if found else "",
                     "raw_matrix_row":found[1][0] if found else "",
                     "raw_matrix_column":found[1][1] if found else "",
                     "name_resolved":found is not None,"score":float(scores[index])})
    write_csv(out / "top20_named_edges.csv", rows)
    diagnostic = {"negative_index_edges":negatives,"unresolved_top20_coordinates":missing,
                  "explanation":"Negative tensor indices are normalised before matching nonzero coordinates."}
    write_json(out / "edge_name_diagnostic.json", diagnostic)
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--scores-run", required=True)
    parser.add_argument("--baseline-dir")
    parser.add_argument("--repo")
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--max-margin-error", type=float, default=0.05)
    opts = parser.parse_args()
    if opts.max_margin_error <= 0 or len(set(opts.random_seeds)) != len(opts.random_seeds):
        parser.error("Use a positive tolerance and distinct random seeds.")
    root = Path(opts.workdir).resolve()
    pilot = Path(opts.scores_run).resolve()
    checkpoint = pilot / "scores.pt"
    original_hash = sha(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    scores = saved["scores"].detach().cpu()
    if saved["args"]["dtype"] != "float32" or scores.ndim != 1 or not torch.isfinite(scores).all():
        raise RuntimeError("Expected finite FP32-run edge scores in a flat vector.")
    adapter_path = root / "code/run_feedback_mattr_fp32.py"
    patch_path = root / "code/run_feedback_mattr_lowmem.py"
    adapter = load_module(adapter_path, "feedback_adapter_readonly")
    if saved["model"] != adapter.MODEL or saved["revision"] != adapter.REVISION:
        raise RuntimeError("Checkpoint model/revision mismatch.")
    arguments = dict(saved["args"])
    arguments.update(workdir=str(root), baseline_dir=opts.baseline_dir or arguments["baseline_dir"],
                     repo=opts.repo or arguments.get("repo"), eval_limit=32, dtype="float32",
                     conversion_only=False, preflight_only=False, cf_cache_gb=0.0,
                     max_margin_error=opts.max_margin_error, max_decision_disagreements=0)
    args = argparse.Namespace(**arguments)
    repo = Path(args.repo).resolve() if args.repo else root / "code/matryoshka-attribution"
    commit = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if commit != (pilot / "repo_commit.txt").read_text().strip():
        raise RuntimeError("Repository commit changed since score training.")
    prepare, preparation_source = prepare_function(adapter, adapter_path)
    patch = load_module(patch_path, "feedback_attention_lowmem")
    patch.self_test()
    patch_manifest = patch.install_patch()
    LOG.info("EVALUATION ONLY: loading saved scores; no training or backward pass will run.")
    model, graph, examples, factory, out, precision = prepare(args)
    write_json(out / "evaluation_arguments.json", vars(opts))
    write_json(out / "attention_patch.json", patch_manifest)
    (out / "preparation_only.py").write_text(preparation_source)
    write_json(out / "evaluation_manifest.json", {
        "source_run":str(pilot),"scores_sha256":original_hash,
        "adapter_sha256":sha(adapter_path),"attention_launcher_sha256":sha(patch_path),
        "official_source_sha256":adapter.EXPECTED_SOURCE_SHA,"repo_commit":commit,
        "dtype":"float32","random_seeds":opts.random_seeds,
        "training_calls":0,"backward_calls":0,
        "notes":"Original preparation loads train prompts only for the existing longest-prompt conversion check. Only dev.jsonl is circuit-evaluated; test data are never read."})
    if len(examples) != 32 or dict(graph.cfg) != saved["graph_cfg"]:
        raise RuntimeError("Development count or graph configuration mismatch.")
    mask = graph.real_edge_mask.bool()
    total = int(mask.sum())
    if total != scores.numel():
        raise RuntimeError("Score-vector length does not match graph.")
    refs = [{"pair_id":e["pair"]["pair_id"],"base":adapter.raw_margin(model,e,"base"),
             "source":adapter.raw_margin(model,e,"source")} for e in examples]
    if not all(math.isfinite(r[side]) for r in refs for side in ("base","source")):
        raise RuntimeError("Nonfinite reference margin")
    write_csv(out / "full_model_reference.csv",refs)
    env = adapter.Environment(factory,model,graph,examples,args)
    device = next(model.parameters()).device
    zeros,ones = torch.zeros(total,device=device),torch.ones(total,device=device)
    endpoints = []
    for i,r in enumerate(refs):
        kept,patched = env.margin(ones,i),env.margin(zeros,i)
        endpoints.append({"pair_id":r["pair_id"],"keep_error":abs(kept-r["base"]),
                          "patch_error":abs(patched-r["source"]),"all_kept":kept,"all_patched":patched})
        write_csv(out / "endpoint_checks.csv",endpoints)
        if not all(math.isfinite(endpoints[-1][k]) and endpoints[-1][k] <= args.max_margin_error
                   for k in ("keep_error","patch_error")):
            raise RuntimeError(f"Intervention endpoint check failed: {r['pair_id']}")
    LOG.info("ENDPOINTS PASSED on all 32 development pairs; no gradient test needed for evaluation.")
    del zeros,ones
    gc.collect()
    torch.cuda.empty_cache()
    old_margins = {(float(r["fraction"]),r["pair_id"]):float(r["margin"])
                   for r in read_csv(pilot / "circuit_margins.csv")}
    old_curve = {float(r["fraction"]):r for r in read_csv(pilot / "curve.csv")}
    all_curves,all_margins,reproduction,summaries = [],[],[],[]
    trial_specs = [("learned",None)] + [(f"random_{seed}",seed) for seed in opts.random_seeds]
    for trial,seed in trial_specs:
        ranking = scores if seed is None else torch.randperm(total,generator=torch.Generator().manual_seed(seed)).float()
        graph.scores[:] = -torch.inf
        graph.scores[mask] = ranking
        curve = []
        for fraction in FRACTIONS:
            k = int(fraction*total)
            graph.apply_topn(k,absolute=False,level="edge",prune=True)
            keep = graph.in_graph[mask].float().to(device)
            current = [env.margin(keep,i) for i in range(len(examples))]
            row = {"trial":trial,"random_seed":seed,"fraction":fraction,"nominal_k":k,
                   "pruned_edges":int(graph.weighted_edge_count()),**metrics(current,refs)}
            curve.append(row)
            all_curves.append(row)
            all_margins.extend({"trial":trial,"fraction":fraction,"pair_id":r["pair_id"],"margin":m}
                               for m,r in zip(current,refs))
            write_csv(out / "curve.csv",all_curves)
            write_csv(out / "circuit_margins.csv",all_margins)
            if seed is None:
                checks = []
                for m,r in zip(current,refs):
                    key = (fraction,r["pair_id"])
                    if key in old_margins:
                        error = abs(m-old_margins[key])
                        checks.append(error)
                        reproduction.append({"fraction":fraction,"pair_id":r["pair_id"],"absolute_error":error})
                write_csv(out / "pilot_reproduction.csv",reproduction)
                if not checks or max(checks) > args.max_margin_error:
                    raise RuntimeError("Saved pilot margins did not reproduce within tolerance.")
                if row["pruned_edges"] != int(old_curve[fraction]["pruned_edges"]):
                    raise RuntimeError("Pilot graph-pruning counts changed.")
            LOG.info("%s %5.1f%%: %d edges; base accuracy %.4f, full %.4f, faithfulness %s",
                     trial,100*fraction,row["pruned_edges"],row["IIA"],row["full_model_base_IIA"],
                     row["faithfulness_nonzero_denominators"])
        summaries.append({"trial":trial,"random_seed":seed,"compactness":compactness(curve)})
        write_csv(out / "trial_summaries.csv",summaries)
    random_curves = [r for r in all_curves if r["trial"] != "learned"]
    if random_curves:
        comparisons = []
        for fraction in FRACTIONS:
            group = [r for r in random_curves if r["fraction"] == fraction]
            learned = next(r for r in all_curves if r["trial"] == "learned" and r["fraction"] == fraction)
            comparisons.append({"fraction":fraction,"learned_IIA":learned["IIA"],
                                "learned_pruned_edges":learned["pruned_edges"],
                                "random_mean_IIA":average([r["IIA"] for r in group]),
                                "random_min_IIA":min(r["IIA"] for r in group),
                                "random_max_IIA":max(r["IIA"] for r in group),
                                "random_mean_pruned_edges":average([r["pruned_edges"] for r in group]),
                                "random_seeds":len(group)})
        write_csv(out / "random_comparison.csv",comparisons)
    if sha(checkpoint) != original_hash or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Checkpoint changed or model gradients were unexpectedly populated.")
    summary = {"model":saved["model"],"revision":saved["revision"],"dtype":"float32",
               "source_run":str(pilot),"source_training_steps":saved["args"]["steps"],
               "new_training_steps":0,"dev_pairs_evaluated":32,"total_scored_edges":total,
               "full_model_base_IIA":average([r["base"]>0 for r in refs]),
               "full_model_source_action_accuracy":average([r["source"]<0 for r in refs]),
               "trial_summaries":summaries,"pilot_max_margin_error":max(r["absolute_error"] for r in reproduction),
               "status":"All requested curve evaluations completed",
               "edge_export_status":"pending",
               "notes":"Base-side circuit evaluation only; same nominal pre-pruning budgets for learned and random rankings. Post-pruning edge counts are reported separately. Faithfulness is the original mean of per-example ratios, with near-zero denominators excluded. No test data or correctness filtering."}
    write_json(out / "summary.json",summary)
    try:
        diagnostic = export_names(graph,scores,out)
        summary["edge_export_status"] = "complete" if not diagnostic["unresolved_top20_coordinates"] else "partial: inspect edge_name_diagnostic.json"
    except Exception as exc:
        LOG.exception("Optional edge-name export failed; completed evaluation summary is already saved.")
        summary["edge_export_status"] = f"failed: {type(exc).__name__}: {exc}"
    write_json(out / "summary.json",summary)
    print("\nEVALUATION-ONLY SUMMARY\n"+json.dumps(summary,indent=2),flush=True)
    print("\nSaved evaluation:",out,flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    main()
