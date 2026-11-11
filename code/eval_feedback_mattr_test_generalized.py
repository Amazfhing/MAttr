"""Held-out test evaluation of a frozen feedback circuit (fully generalized).
Loads fixed_mask.pt directly: no score fitting, ranking, top-k selection, or pruning.
Accepts any workload version and mask that matches the reverse evaluator's saved mask identity.
No hard-coded edge counts, pair counts, or mask hashes - all inferred from circuit run metadata.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import torch

LOG = logging.getLogger("feedback_frozen_test_v2")


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def write_csv(path,rows):
    if not rows:
        return
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def average(values):
    return sum(values)/len(values) if values else None


def metrics(current,refs):
    if not all(math.isfinite(m) for m in current):
        raise RuntimeError("Nonfinite circuit margins")
    faith=[(m-r["source"])/(r["base"]-r["source"])
           for m,r in zip(current,refs) if abs(r["base"]-r["source"])>1e-6]
    agreement=[(m>0)==(r["base"]>0) for m,r in zip(current,refs) if r["base"]!=0]
    return {"IIA":average([m>0 for m in current]),
            "full_model_base_IIA":average([r["base"]>0 for r in refs]),
            "agreement_with_full_non_ties":average(agreement),
            "faithfulness_nonzero_denominators":average(faith),
            "faithfulness_defined_examples":len(faith),
            "mean_circuit_margin":average(current),
            "mean_absolute_margin_error_vs_full":average([abs(m-r["base"]) for m,r in zip(current,refs)]),
            "zero_circuit_margins":sum(m==0 for m in current),"evaluation_pairs":len(refs)}


def reverse_example(example):
    original=example["pair"]
    pair=dict(original)
    for stem in ("messages","label","action"):
        b,s="base_"+stem,"source_"+stem
        if b in original or s in original:
            if b not in original or s not in original:
                raise RuntimeError("Unpaired base/source metadata")
            pair[b],pair[s]=original[s],original[b]
    base,source=example["tokens"]["source"],example["tokens"]["base"]
    if base.shape!=source.shape:
        raise RuntimeError("Unequal token lengths")
    a,b=example["pretok"][-1]
    if a==b:
        raise RuntimeError("Expected distinct target labels")
    return {"pair":pair,"tokens":{"base":base,"source":source},
            "pretok":(base,torch.ones_like(base),base.shape[1],source,(b,a))}


def close(actual, expected, tolerance, description):
    error = abs(actual-expected)
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(f"{description}: actual={actual}, expected={expected}, error={error}")
    return error


def report(current, refs, utility):
    return {"contexts":len(current),"circuit_correct":sum(m > 0 for m in current),
            "full_model_correct":sum(r["base"] > 0 for r in refs),**utility.metrics(current,refs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir",required=True)
    parser.add_argument("--circuit-run",required=True)
    parser.add_argument("--max-margin-error",type=float,default=.05)
    parser.add_argument("--data-version",default=None,
                       help="Workload version (e.g. 'v1', 'v2'). If omitted, auto-detected from circuit run metadata.")
    options = parser.parse_args()
    if options.max_margin_error <= 0:
        parser.error("A positive numerical tolerance is required")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one authorised GPU with CUDA_VISIBLE_DEVICES")
    root = Path(options.workdir).resolve()
    circuit_run = Path(options.circuit_run).resolve()
    code = root / "code"
    utility = SimpleNamespace(sha=sha,write_json=write_json,write_csv=write_csv,metrics=metrics)
    adapter = import_file(code / "run_feedback_mattr_fp32.py","feedback_test_adapter")
    reverse_api = SimpleNamespace(reverse_example=reverse_example)

    prior = json.loads((circuit_run / "summary.json").read_text())
    if (prior["model"] != adapter.MODEL or prior["revision"] != adapter.REVISION
        or prior["dtype"] != "float32" or prior["nominal_fraction"] != .1):
        raise RuntimeError("Expected FP32 nominal-10% circuit")

    # Auto-detect data version and name from circuit run or use explicit override
    if options.data_version:
        data_version = options.data_version
    else:
        # Trace back to source run to find baseline, then extract version
        source_run = Path(prior["source_run"]).resolve()
        source_args = json.loads((source_run / "arguments.json").read_text())
        baseline = json.loads((Path(source_args["baseline_dir"]) / "summary.json").read_text())
        data_version = baseline.get("version", "v1")

    # Auto-detect data name from baseline
    source_run = Path(prior["source_run"]).resolve()
    source_args = json.loads((source_run / "arguments.json").read_text())
    baseline = json.loads((Path(source_args["baseline_dir"]) / "summary.json").read_text())
    data_name = baseline.get("data_name", "feedback_retry_answer")

    LOG.info("Auto-detected workload: %s version %s", data_name, data_version)

    # Determine generator file based on data name and version
    if data_name == "feedback_retry_answer":
        if data_version == "v1":
            generator_file = "feedback_pilot.py"
        else:
            generator_file = f"feedback_pilot_{data_version}.py"
    elif data_name == "export_goal_verification":
        generator_file = "export_goal_workload.py"
    else:
        # Generic: try {data_name}_workload.py or {data_name}_{version}.py
        candidate1 = f"{data_name}_workload.py"
        candidate2 = f"{data_name}_{data_version}.py"
        if (code / candidate1).exists():
            generator_file = candidate1
        elif (code / candidate2).exists():
            generator_file = candidate2
        else:
            raise RuntimeError(f"Cannot find generator for {data_name} {data_version}. Tried {candidate1}, {candidate2}")

    if not (code / generator_file).exists():
        raise RuntimeError(f"Generator {generator_file} not found for {data_name} version {data_version}")

    generator = import_file(code / generator_file,"feedback_generator_readonly")

    # Flexible edge count and mask verification - no hard-coded values
    mask_path = circuit_run / "fixed_mask.pt"
    mask_file_sha = utility.sha(mask_path)
    frozen = torch.load(mask_path,map_location="cpu",weights_only=True)
    keep_cpu = frozen["keep_mask"].detach().cpu().clone()
    if keep_cpu.dtype != torch.bool or keep_cpu.ndim != 1:
        raise RuntimeError("Expected the saved one-dimensional Boolean mask")
    mask_sha = hashlib.sha256(keep_cpu.numpy().tobytes()).hexdigest()

    # Verify mask matches reverse evaluator's saved identity (no hard-coded hash)
    if mask_sha != frozen["mask_sha256"] or mask_sha != prior["fixed_mask_sha256"]:
        raise RuntimeError("Saved mask does not match the development-selected candidate")

    retained_edges = int(keep_cpu.sum())
    if retained_edges != prior["retained_edges"]:
        raise RuntimeError("Saved retained-edge count changed")

    LOG.info("TEST PROTOCOL: %s %s workload, %d retained edges, mask SHA256 %s",
             data_name, data_version, retained_edges, mask_sha[:16])

    source_run = Path(prior["source_run"]).resolve()
    if utility.sha(source_run / "scores.pt") != frozen["scores_sha256"]:
        raise RuntimeError("Source extraction checkpoint changed")
    source_args = json.loads((source_run / "arguments.json").read_text())
    repo = Path(source_args["repo"]).resolve() if source_args.get("repo") else code / "matryoshka-attribution"
    commit = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if commit != (source_run / "repo_commit.txt").read_text().strip() or commit != (circuit_run / "repo_commit.txt").read_text().strip():
        raise RuntimeError("Repository commit changed")

    # Flexible test data path and pair count
    test_path = root / f"data/{data_name}_{data_version}/test.jsonl"
    test_sha = utility.sha(test_path)
    records = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]

    # Infer expected pair count from actual test data
    expected_pairs = len(records)
    generator_pairs = generator.build_pairs("test", expected_pairs)

    if records != generator_pairs:
        raise RuntimeError("Test dataset differs from the audited generator; no data are overwritten")
    if len({r["pair_id"] for r in records}) != expected_pairs or any(r["split"] != "test" for r in records):
        raise RuntimeError(f"Expected {expected_pairs} distinct held-out test pairs")

    results_root = root / "results"
    results_root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"feedback_mattr_test_{options.data_version}_",dir=str(results_root)))
    LOG.info("TEST OUTPUT DIRECTORY: %s",out)

    expected_contexts = expected_pairs * 2
    protocol = {"split":"test","pairs":expected_pairs,"contexts":expected_contexts,
                "data_version":data_version,
                "orientations":["original","reversed"],"dtype":"float32",
                "candidate_mask_sha256":mask_sha,"candidate_retained_edges":retained_edges,
                "new_training_steps":0,"ranking_or_pruning_calls":0,
                "metric":"Pairwise A/B target-logit preference; ties count as incorrect",
                "full_vocabulary_argmax_measured":False,"correctness_filtering":False,
                "unequal_length_policy":"Stop rather than filter or change intervention alignment",
                "faithfulness":"Original mean of per-pair normalised margin ratios; exclude |denominator| <= 1e-6 and report counts",
                "max_margin_error":options.max_margin_error,
                "notes":f"{data_version} workload test evaluation. No mask or budget selection uses test outcomes. Dataset generator is imported only for read-only identity verification."}
    utility.write_json(out / "protocol.json",protocol)

    paths = {"test_data":test_path,"mask_file":mask_path,"generator":code / generator_file,
             "adapter":code / "run_feedback_mattr_fp32.py","attention_patch":code / "run_feedback_mattr_lowmem.py",
             "test_launcher":Path(__file__)}
    utility.write_json(out / "manifest.json",{
        "circuit_run":str(circuit_run),"source_run":str(source_run),"model":adapter.MODEL,
        "revision":adapter.REVISION,"repo_commit":commit,
        "paths":{k:str(v) for k,v in paths.items()},"sha256":{k:utility.sha(v) for k,v in paths.items()},
        "source_scores_sha256":frozen["scores_sha256"],"mask_sha256":mask_sha,
        "torch_version":torch.__version__,"official_source_sha256":adapter.EXPECTED_SOURCE_SHA})
    (out / "repo_commit.txt").write_text(commit+"\n")

    factory = adapter.official_factory(repo / "scripts/mib/eval_mib_edge.py",out)
    baseline = json.loads((Path(source_args["baseline_dir"]) / "summary.json").read_text())
    if baseline["model"] != adapter.MODEL or baseline["resolved_revision"] != adapter.REVISION:
        raise RuntimeError("Pinned model snapshot metadata mismatch")
    snapshot = Path(baseline["snapshot"])
    if not snapshot.is_dir():
        raise RuntimeError("Pinned local snapshot is missing")

    patch = import_file(code / "run_feedback_mattr_lowmem.py","feedback_test_attention_patch")
    patch.self_test()
    utility.write_json(out / "attention_patch.json",patch.install_patch())

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from matryoshka_attribution.deps import add_mib_to_sys_path
    add_mib_to_sys_path()
    from transformer_lens import HookedTransformer
    from eap.graph import Graph

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    label_tokens = {s:tokenizer.encode(s,add_special_tokens=False) for s in ("A","B")}
    if any(len(ids) != 1 for ids in label_tokens.values()):
        raise RuntimeError("Labels must remain single tokens")
    labels = {s:ids[0] for s,ids in label_tokens.items()}
    if labels != baseline["label_ids"]:
        raise RuntimeError("Label token IDs changed")

    device = torch.device("cuda:0")
    examples = adapter.read_pairs(test_path,tokenizer,labels,device)
    if [e["pair"] for e in examples] != records:
        raise RuntimeError("Test data changed during tokenisation")

    LOG.info("Loaded exactly %d TEST pairs; no training or development dataset loaded.", expected_pairs)
    LOG.info("Loading the pinned HF checkpoint in FP32 for test-only numerical comparison.")

    hf = AutoModelForCausalLM.from_pretrained(str(snapshot),dtype=torch.float32,
        attn_implementation="eager",local_files_only=True).to(device).eval()
    hf.requires_grad_(False)
    hf_refs = {(e["pair"]["pair_id"],side):adapter.raw_margin(hf,e,side,hf=True)
               for e in examples for side in ("base","source")}
    hf.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    LOG.info("Converting the same checkpoint; no folding or centering.")
    model = HookedTransformer.from_pretrained(adapter.MODEL,hf_model=hf,tokenizer=tokenizer,
        revision=adapter.REVISION,device="cuda",dtype=torch.float32,
        fold_ln=False,center_writing_weights=False,center_unembed=False,
        fold_value_biases=False,refactor_factored_attn_matrices=False)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    model.requires_grad_(False)
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    comparison,lookup = [],{}
    for e in examples:
        for side in ("base","source"):
            key = (e["pair"]["pair_id"],side)
            actual,reference = adapter.raw_margin(model,e,side),hf_refs[key]
            if not math.isfinite(actual) or not math.isfinite(reference):
                raise RuntimeError("Nonfinite full-model margin")
            lookup[key] = actual
            comparison.append({"pair_id":key[0],"side":side,"HF_margin":reference,
                               "TL_margin":actual,"absolute_error":abs(actual-reference),
                               "decision_disagrees":actual*reference < 0,
                               "either_margin_is_zero":actual == 0 or reference == 0})
    utility.write_csv(out / "hf_tl_comparison.csv",comparison)
    max_error = max(r["absolute_error"] for r in comparison)
    disagreements = sum(r["decision_disagrees"] for r in comparison)
    utility.write_json(out / "precision_comparison.json",{
        "contexts":expected_contexts,"max_margin_error":max_error,"strict_preference_reversals":disagreements})
    if max_error > options.max_margin_error or disagreements:
        raise RuntimeError("HF/TL parity failed; circuit test evaluation not started")
    LOG.info("TEST HF/TL parity passed: max error %.8g; %d preference reversals",max_error,disagreements)

    graph = Graph.from_model(model)
    real = graph.real_edge_mask.bool()
    if dict(graph.cfg) != frozen["graph_cfg"] or int(real.sum()) != keep_cpu.numel():
        raise RuntimeError("Saved mask and current graph do not match")
    if hashlib.sha256(real.numpy().tobytes()).hexdigest() != frozen["real_edge_mask_sha256"]:
        raise RuntimeError("Graph edge ordering changed")

    graph.in_graph[real] = keep_cpu
    keep = keep_cpu.float().to(device)
    refs = [{"pair_id":e["pair"]["pair_id"],"base":lookup[(e["pair"]["pair_id"],"base")],
             "source":lookup[(e["pair"]["pair_id"],"source")]} for e in examples]
    reverse_refs = [{"pair_id":r["pair_id"],"base":-r["source"],"source":-r["base"]} for r in refs]
    reverse_examples = [reverse_api.reverse_example(e) for e in examples]
    utility.write_csv(out / "full_model_reference.csv",refs)
    utility.write_csv(out / "reversed_full_model_reference.csv",reverse_refs)

    env_args = SimpleNamespace(cf_cache_gb=0.)
    original_env = adapter.Environment(factory,model,graph,examples,env_args)
    reversed_env = adapter.Environment(factory,model,graph,reverse_examples,env_args)
    zeros,ones = torch.zeros(keep_cpu.numel(),device=device),torch.ones(keep_cpu.numel(),device=device)

    endpoints=[]
    for direction,env,direction_refs in (("original",original_env,refs),("reversed",reversed_env,reverse_refs)):
        for i,r in enumerate(direction_refs):
            kept,patched = env.margin(ones,i),env.margin(zeros,i)
            keep_error = close(kept,r["base"],options.max_margin_error,f"{direction} all-kept endpoint")
            patch_error = close(patched,r["source"],options.max_margin_error,f"{direction} all-patched endpoint")
            endpoints.append({"direction":direction,"pair_id":r["pair_id"],"all_kept":kept,
                              "expected_base":r["base"],"all_patched":patched,"expected_source":r["source"],
                              "keep_error":keep_error,"patch_error":patch_error})
            utility.write_csv(out / "endpoint_checks.csv",endpoints)
        LOG.info("TEST %s endpoints passed on all %d pairs",direction,expected_pairs)
    del zeros,ones

    values={"original":[],"reversed":[]}
    margins=[]
    for direction,env,direction_refs in (("original",original_env,refs),("reversed",reversed_env,reverse_refs)):
        for i,r in enumerate(direction_refs):
            margin = env.margin(keep,i)
            if not math.isfinite(margin):
                raise RuntimeError("Nonfinite circuit margin")
            values[direction].append(margin)
            margins.append({"direction":direction,"pair_id":r["pair_id"],"circuit_margin":margin,
                            "full_model_margin":r["base"],"donor_full_model_margin":r["source"],
                            "circuit_correct":margin > 0,"full_model_correct":r["base"] > 0})
            utility.write_csv(out / "circuit_margins.csv",margins)
            if (i+1)%16 == 0:
                LOG.info("TEST %s %d/%d: %d correct so far",direction,i+1,expected_pairs,sum(m > 0 for m in values[direction]))

    original,reversed_values = values["original"],values["reversed"]
    paired=[]
    circuit_actions,full_actions=defaultdict(list),defaultdict(list)
    for e,r,rr,a,b in zip(examples,refs,reverse_refs,original,reversed_values):
        pair=e["pair"]
        paired.append({"pair_id":pair["pair_id"],"original_target_action":pair["base_action"],
                       "reversed_target_action":pair["source_action"],"original_circuit_margin":a,
                       "reversed_circuit_margin":b,"original_full_margin":r["base"],
                       "reversed_full_margin":rr["base"],"original_correct":a > 0,"reversed_correct":b > 0,
                       "both_correct":a > 0 and b > 0,"full_both_correct":r["base"] > 0 and rr["base"] > 0})
        for action,cm,fm in ((pair["base_action"],a,r["base"]),(pair["source_action"],b,rr["base"])):
            circuit_actions[action].append(cm > 0)
            full_actions[action].append(fm > 0)
    utility.write_csv(out / "paired_results.csv",paired)

    if (utility.sha(mask_path) != mask_file_sha or utility.sha(test_path) != test_sha
        or not torch.equal(keep.cpu().bool(),keep_cpu) or any(p.grad is not None for p in model.parameters())):
        raise RuntimeError("Mask/data changed or unexpected gradients appeared")

    source_action_accuracy=sum(r["source"] < 0 for r in refs)/expected_pairs
    full_both=sum(r["full_both_correct"] for r in paired)
    both=sum(r["both_correct"] for r in paired)

    summary={"model":adapter.MODEL,"revision":adapter.REVISION,"dtype":"float32","split":"test",
             "data_version":data_version,"circuit_run":str(circuit_run),
             "new_training_steps":0,"ranking_or_pruning_calls":0,
             "test_pairs_evaluated":expected_pairs,"total_scored_edges":keep_cpu.numel(),
             "retained_edges":int(keep_cpu.sum()),"fixed_mask_sha256":mask_sha,
             "original_direction":report(original,refs,utility),
             "reversed_direction":report(reversed_values,reverse_refs,utility),
             "paired":{"pairs":expected_pairs,"circuit_both_sides_correct":both,
                       "circuit_both_sides_accuracy":both/expected_pairs,
                       "full_model_both_sides_correct":full_both,
                       "full_model_both_sides_accuracy":full_both/expected_pairs},
             "combined_context_accuracy":(sum(m > 0 for m in original)+sum(m > 0 for m in reversed_values))/expected_contexts,
             "full_model_combined_context_accuracy":(sum(r["base"] > 0 for r in refs)/expected_pairs+source_action_accuracy)/2,
             "per_target_action":{action:{"contexts":len(bits),"circuit_correct":sum(bits),
                                          "circuit_accuracy":sum(bits)/len(bits),
                                          "full_model_correct":sum(full_actions[action]),
                                          "full_model_accuracy":sum(full_actions[action])/len(bits)}
                                  for action,bits in sorted(circuit_actions.items())},
             "parity_max_margin_error":max_error,"strict_preference_reversals":disagreements,
             "endpoint_max_margin_error":max(max(r["keep_error"],r["patch_error"]) for r in endpoints),
             "status":"Held-out evaluation completed with unchanged development-selected mask",
             "notes":f"{data_version} workload; both input orientations; no test-based selection or filtering. Only pairwise A/B logit preference is measured, not native tool execution or full-vocabulary generation accuracy. Saved mask loaded directly; compatible graph structure is rebuilt for hooks, but no scores are ranked and no circuit is re-pruned."}
    utility.write_json(out / "summary.json",summary)
    print(f"\nHELD-OUT FIXED-MASK TEST SUMMARY ({data_version})\n"+json.dumps(summary,indent=2),flush=True)
    print("\nSaved held-out evaluation:",out,flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    main()
