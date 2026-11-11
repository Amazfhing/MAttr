"""Evaluate a single saved feedback circuit in both input-pair orientations.
No training, backward passes, random-ranking sweep, or test-set access.
Requires the existing adapter, low-memory patch, and saved-score evaluator.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import importlib.util

import torch

LOG = logging.getLogger("feedback_fixed_mask_reverse")


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reverse_example(example):
    original = example["pair"]
    pair = dict(original)
    for stem in ("messages", "label", "action"):
        b, s = "base_" + stem, "source_" + stem
        if b in original or s in original:
            if b not in original or s not in original:
                raise RuntimeError(f"Unpaired metadata fields: {b}, {s}")
            pair[b], pair[s] = original[s], original[b]
    base, source = example["tokens"]["source"], example["tokens"]["base"]
    if base.shape != source.shape:
        raise RuntimeError("Unequal pair lengths")
    labels = example["pretok"][-1]
    if labels[0] == labels[1]:
        raise RuntimeError("Expected distinct paired target labels")
    return {"pair":pair,"tokens":{"base":base,"source":source},
            "pretok":(base,torch.ones_like(base),base.shape[1],source,(labels[1],labels[0]))}


def close(actual, expected, tolerance, description):
    error = abs(actual-expected)
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(f"{description}: actual={actual}, expected={expected}, error={error}")
    return error


def direction_report(current, refs, utility):
    return {"contexts":len(current),"circuit_correct":sum(m > 0 for m in current),
            "full_model_correct":sum(r["base"] > 0 for r in refs),
            **utility.metrics(current,refs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir",required=True)
    parser.add_argument("--scores-run",required=True)
    parser.add_argument("--reference-eval-run",required=True)
    parser.add_argument("--fraction",type=float,default=.1)
    parser.add_argument("--max-margin-error",type=float,default=.05)
    options = parser.parse_args()
    if not 0 < options.fraction <= 1 or options.max_margin_error <= 0:
        parser.error("Require 0 < fraction <= 1 and a positive tolerance.")
    root = Path(options.workdir).resolve()
    training_run = Path(options.scores_run).resolve()
    reference_run = Path(options.reference_eval_run).resolve()
    code = root / "code"
    utility_path = code / "eval_feedback_mattr_saved.py"
    utility = import_file(utility_path,"feedback_saved_evaluation_api")
    adapter_path = code / "run_feedback_mattr_fp32.py"
    adapter = import_file(adapter_path,"feedback_original_adapter_api")
    checkpoint_path = training_run / "scores.pt"
    original_hash = utility.sha(checkpoint_path)
    saved = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    scores = saved["scores"].detach().cpu()
    if saved["model"] != adapter.MODEL or saved["revision"] != adapter.REVISION:
        raise RuntimeError("Checkpoint model/revision mismatch")
    if saved["args"]["dtype"] != "float32" or scores.ndim != 1 or not torch.isfinite(scores).all():
        raise RuntimeError("Expected finite scores from the original FP32 run")
    reference_summary = json.loads((reference_run / "summary.json").read_text())
    if (Path(reference_summary["source_run"]).resolve() != training_run
        or reference_summary["model"] != saved["model"]
        or reference_summary["revision"] != saved["revision"]
        or reference_summary["dev_pairs_evaluated"] != 32):
        raise RuntimeError("Reference evaluation does not match this extraction run")
    curves = [r for r in utility.read_csv(reference_run / "curve.csv")
              if r["trial"] == "learned" and abs(float(r["fraction"])-options.fraction) < 1e-12]
    if len(curves) != 1:
        raise RuntimeError("Budget must match exactly one previously evaluated learned curve point")
    reference_curve = curves[0]
    reference_margins = {}
    for row in utility.read_csv(reference_run / "circuit_margins.csv"):
        if row["trial"] == "learned" and abs(float(row["fraction"])-options.fraction) < 1e-12:
            if row["pair_id"] in reference_margins:
                raise RuntimeError("Duplicate reference pair ID")
            reference_margins[row["pair_id"]] = float(row["margin"])
    if len(reference_margins) != 32:
        raise RuntimeError("Expected 32 saved reference margins")
    values = dict(saved["args"])
    values.update(workdir=str(root),eval_limit=32,dtype="float32",conversion_only=False,
                  preflight_only=False,cf_cache_gb=0.,max_margin_error=options.max_margin_error,
                  max_decision_disagreements=0)
    args = argparse.Namespace(**values)
    repo = Path(args.repo).resolve() if args.repo else code / "matryoshka-attribution"
    commit = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if commit != (training_run / "repo_commit.txt").read_text().strip():
        raise RuntimeError("Repository commit changed since extraction")
    prepare, preparation_source = utility.prepare_function(adapter,adapter_path)
    patch_path = code / "run_feedback_mattr_lowmem.py"
    patch = import_file(patch_path,"feedback_reverse_attention_patch")
    patch.self_test()
    patch_manifest = patch.install_patch()
    LOG.info("FIXED-MASK BIDIRECTIONAL EVALUATION: no training or random sweep.")
    model,graph,examples,factory,out,precision = prepare(args)
    utility.write_json(out / "reverse_arguments.json",vars(options))
    utility.write_json(out / "attention_patch.json",patch_manifest)
    (out / "preparation_only.py").write_text(preparation_source)
    utility.write_json(out / "reverse_manifest.json",{
        "evaluation_kind":"fixed-mask input-pair reversal",
        "source_run":str(training_run),"reference_evaluation":str(reference_run),
        "source_scores_sha256":original_hash,"adapter_sha256":utility.sha(adapter_path),
        "evaluation_api_sha256":utility.sha(utility_path),"reverse_launcher_sha256":utility.sha(Path(__file__)),
        "attention_patch_sha256":utility.sha(patch_path),"repo_commit":commit,
        "training_calls":0,"backward_calls":0,"fraction":options.fraction,
        "notes":"Same retained edges in both orientations. Source/base inputs AND target-label ordering are exchanged. Test data are never loaded."})
    ids = {e["pair"]["pair_id"] for e in examples}
    if len(examples) != 32 or len(ids) != 32 or ids != set(reference_margins):
        raise RuntimeError("Development pairs changed or contain duplicate IDs")
    if dict(graph.cfg) != saved["graph_cfg"]:
        raise RuntimeError("Graph configuration changed")
    real = graph.real_edge_mask.bool()
    total = int(real.sum())
    if scores.numel() != total:
        raise RuntimeError("Score vector and graph do not match")
    graph.scores[:] = -torch.inf
    graph.scores[real] = scores
    graph.apply_topn(int(options.fraction*total),absolute=False,level="edge",prune=True)
    keep_cpu = graph.in_graph[real].detach().cpu().clone().bool()
    kept_count = int(keep_cpu.sum())
    if kept_count != int(reference_curve["pruned_edges"]) or kept_count != int(graph.weighted_edge_count()):
        raise RuntimeError("Reconstructed circuit size does not match previous evaluation")
    mask_hash = hashlib.sha256(keep_cpu.numpy().tobytes()).hexdigest()
    real_hash = hashlib.sha256(real.numpy().tobytes()).hexdigest()
    torch.save({"keep_mask":keep_cpu,"graph_cfg":dict(graph.cfg),"fraction":options.fraction,
                "scores_sha256":original_hash,"mask_sha256":mask_hash,
                "real_edge_mask_sha256":real_hash},out / "fixed_mask.pt")
    LOG.info("FROZEN CIRCUIT: %d / %d edges at nominal %.1f%%; mask SHA256 %s",
             kept_count,total,100*options.fraction,mask_hash)
    parity = {}
    for row in utility.read_csv(out / "hf_tl_comparison.csv"):
        if row["pair_id"] in ids:
            key = (row["pair_id"],row["side"])
            if key in parity:
                raise RuntimeError("Duplicate conversion-reference ID")
            parity[key] = float(row["TL_margin"])
    if len(parity) != 64:
        raise RuntimeError("Missing development full-model references")
    refs = [{"pair_id":e["pair"]["pair_id"],"base":parity[(e["pair"]["pair_id"],"base")],
             "source":parity[(e["pair"]["pair_id"],"source")]} for e in examples]
    reversed_examples = [reverse_example(e) for e in examples]
    reverse_refs = [{"pair_id":r["pair_id"],"base":-r["source"],"source":-r["base"]} for r in refs]
    utility.write_csv(out / "full_model_reference.csv",refs)
    utility.write_csv(out / "reversed_full_model_reference.csv",reverse_refs)
    original_env = adapter.Environment(factory,model,graph,examples,args)
    reversed_env = adapter.Environment(factory,model,graph,reversed_examples,args)
    device = next(model.parameters()).device
    keep = keep_cpu.float().to(device)
    original_values,reproduction = [],[]
    for i,e in enumerate(examples):
        current = original_env.margin(keep,i)
        original_values.append(current)
        error = close(current,reference_margins[e["pair"]["pair_id"]],options.max_margin_error,
                      "Original-direction reproduction failed")
        reproduction.append({"pair_id":e["pair"]["pair_id"],"margin":current,"absolute_error":error})
        utility.write_csv(out / "original_reproduction.csv",reproduction)
    LOG.info("ORIGINAL DIRECTION REPRODUCED: max margin error %.8g",
             max(r["absolute_error"] for r in reproduction))
    zeros,ones = torch.zeros(total,device=device),torch.ones(total,device=device)
    endpoints = []
    for i,r in enumerate(reverse_refs):
        kept,patched = reversed_env.margin(ones,i),reversed_env.margin(zeros,i)
        keep_error = close(kept,r["base"],options.max_margin_error,"Reversed all-kept endpoint failed")
        patch_error = close(patched,r["source"],options.max_margin_error,"Reversed all-patched endpoint failed")
        endpoints.append({"pair_id":r["pair_id"],"all_kept":kept,"expected_base":r["base"],
                          "all_patched":patched,"expected_source":r["source"],
                          "keep_error":keep_error,"patch_error":patch_error})
        utility.write_csv(out / "reversed_endpoint_checks.csv",endpoints)
    LOG.info("REVERSED ENDPOINTS PASSED on all 32 pairs; evaluating unchanged candidate mask.")
    del zeros,ones
    reversed_values,paired_rows = [],[]
    by_action = defaultdict(list)
    for i,(e,r,rr) in enumerate(zip(examples,refs,reverse_refs)):
        forward = original_values[i]
        backward = reversed_env.margin(keep,i)
        if not math.isfinite(backward):
            raise RuntimeError("Nonfinite reversed circuit margin")
        reversed_values.append(backward)
        original_action = e["pair"].get("base_action","unknown")
        reversed_action = e["pair"].get("source_action","unknown")
        by_action[original_action].append(forward > 0)
        by_action[reversed_action].append(backward > 0)
        paired_rows.append({"pair_id":r["pair_id"],"original_target_action":original_action,
                           "reversed_target_action":reversed_action,
                           "original_circuit_margin":forward,"original_full_margin":r["base"],
                           "reversed_circuit_margin":backward,"reversed_full_margin":rr["base"],
                           "original_correct":forward > 0,"reversed_correct":backward > 0,
                           "both_correct":forward > 0 and backward > 0,
                           "full_original_correct":r["base"] > 0,"full_reversed_correct":rr["base"] > 0})
        utility.write_csv(out / "paired_results.csv",paired_rows)
        if (i+1)%8 == 0:
            LOG.info("Reversed evaluation %d/32: %d correct so far",i+1,sum(v > 0 for v in reversed_values))
    if not torch.equal(keep.detach().cpu().bool(),keep_cpu):
        raise RuntimeError("Fixed mask changed")
    if utility.sha(checkpoint_path) != original_hash or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Checkpoint changed or unexpected model gradients")
    both = sum(a > 0 and b > 0 for a,b in zip(original_values,reversed_values))
    full_both = sum(r["base"] > 0 and rr["base"] > 0 for r,rr in zip(refs,reverse_refs))
    summary = {"model":saved["model"],"revision":saved["revision"],"dtype":"float32",
               "source_run":str(training_run),"reference_evaluation":str(reference_run),
               "new_training_steps":0,"dev_pairs_evaluated":32,"nominal_fraction":options.fraction,
               "total_scored_edges":total,"retained_edges":kept_count,"fixed_mask_sha256":mask_hash,
               "original_direction":direction_report(original_values,refs,utility),
               "reversed_direction":direction_report(reversed_values,reverse_refs,utility),
               "paired":{"pairs":32,"circuit_both_sides_correct":both,"circuit_both_sides_accuracy":both/32,
                         "full_model_both_sides_correct":full_both,"full_model_both_sides_accuracy":full_both/32},
               "combined_context_accuracy":(sum(m > 0 for m in original_values)+sum(m > 0 for m in reversed_values))/64,
               "per_target_action":{action:{"contexts":len(v),"correct":sum(v),"accuracy":sum(v)/len(v)}
                                    for action,v in sorted(by_action.items())},
               "original_reproduction_max_margin_error":max(r["absolute_error"] for r in reproduction),
               "reversed_endpoint_max_margin_error":max(max(r["keep_error"],r["patch_error"]) for r in endpoints),
               "status":"Both orientations evaluated with the same frozen mask",
               "notes":"All development pairs included, including full-model errors. This reverses input/label roles, not the kept-edge mask; it is not top-k noising and is not a held-out test evaluation."}
    utility.write_json(out / "summary.json",summary)
    print("\nBIDIRECTIONAL FIXED-MASK SUMMARY\n"+json.dumps(summary,indent=2),flush=True)
    print("\nSaved bidirectional evaluation:",out,flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    main()
