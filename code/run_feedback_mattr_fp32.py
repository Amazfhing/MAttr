
import argparse
import ast
import copy
import csv
import gc
import hashlib
import inspect
import json
import logging
import math
from pathlib import Path
import random
import subprocess
import tempfile

import torch
from torch.utils.checkpoint import checkpoint
from einops import einsum

EXPECTED_SOURCE_SHA = "042b354cbace9c509d339b975557fbb45821639ece50aa4870097f3971de2db1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feedback_mattr")


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def official_factory(path, out):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Official source differs from the uploaded version. Stopping rather than adapting unknown code.")
    tree = ast.parse(raw.decode())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    def assignment(n, name):
        return isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    start = next(n for n in main.body if assignment(n, "source_hooks"))
    stop = next(n for n in main.body if assignment(n, "n_examples"))
    capture = [copy.deepcopy(n) for n in main.body if start.lineno <= n.lineno < stop.lineno]
    loss = copy.deepcopy(next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "loss_fn"))
    changes = 0
    for node in loss.body:
        if assignment(node, "idx"):
            node.value = ast.Call(func=ast.Name(id="pick_index", ctx=ast.Load()), args=[], keywords=[])
            changes += 1
    assert changes == 1
    factory = ast.parse("def factory(model, graph, pretok, args, pick_index):\n    pass\n").body[0]
    setup = ast.parse("""
device = next(model.parameters()).device
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
n_examples = len(pretok)
real_flat = graph.real_edge_mask.flatten().bool()
cumsum = (real_flat.float().cumsum(0).long() - 1).clamp(min=0).to(device)
real_flat_device = real_flat.float().to(device)
corrupt_topk = False
corrupted_cache = {}
cc_bytes = 0
cc_max_bytes = int(args.cf_cache_gb * 1e9)
cc_stats = {"seen": 0, "fwd": 0, "full_logged": False}
""").body
    factory.body = setup + capture + [loss, ast.Return(value=ast.Name(id="loss_fn", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    (out / "reused_official_edge_factory.py").write_text(ast.unparse(module))
    namespace = {"torch": torch, "checkpoint": checkpoint, "einsum": einsum, "logger": logger}
    exec(compile(module, str(path) + ":task_adapter", "exec"), namespace)
    return namespace["factory"]


class Environment:
    def __init__(self, factory, model, graph, examples, args):
        self.forced = None
        self.examples = examples
        self.loss = factory(model, graph, [e["pretok"] for e in examples], args, self.pick)

    def pick(self):
        return random.randrange(len(self.examples)) if self.forced is None else self.forced

    def fixed_loss(self, mask, index):
        old = self.forced
        self.forced = index
        try:
            value = self.loss(mask)
            if value is None:
                raise RuntimeError("Unexpected unequal-length pair.")
            return value
        finally:
            self.forced = old

    def margin(self, mask, index):
        with torch.no_grad():
            return -self.fixed_loss(mask, index).item()


def read_pairs(path, tokenizer, label_ids, device):
    pairs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    examples = []
    for pair in pairs:
        sides = {}
        for side in ["base", "source"]:
            text = tokenizer.apply_chat_template(pair[f"{side}_messages"], tokenize=False,
                                                add_generation_prompt=True)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            sides[side] = torch.tensor([token_ids], dtype=torch.long, device=device)
        if sides["base"].shape != sides["source"].shape:
            raise RuntimeError(f"Unequal token lengths: {pair['pair_id']}; no silent filtering.")
        labels = (label_ids[pair["base_label"]], label_ids[pair["source_label"]])
        mask = torch.ones_like(sides["base"])
        examples.append({"pair": pair, "tokens": sides,
            "pretok": (sides["base"], mask, sides["base"].shape[1], sides["source"], labels)})
    return examples


def raw_margin(model, example, side, hf=False):
    tokens = example["tokens"][side]
    with torch.no_grad():
        if hf:
            logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        else:
            logits = model(tokens, attention_mask=torch.ones_like(tokens), prepend_bos=False)
        a, b = example["pretok"][-1]
        return (logits[0, -1, a].float() - logits[0, -1, b].float()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--n-iters", type=int, default=30)
    parser.add_argument("--cf-cache-gb", type=float, default=0.0)
    parser.add_argument("--max-margin-error", type=float, default=0.05)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--conversion-only", action="store_true")
    parser.add_argument("--max-decision-disagreements", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--data-version", choices=["v1", "v2"], default=None)
    parser.add_argument("--data-name", default=None, help="Dataset folder name (e.g., 'export_goal_verification'). If not specified, uses baseline's dataset name.")
    parser.add_argument("--skip-count-check", action="store_true", help="Skip train/dev count assertions")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_limit < 1:
        parser.error("steps and eval-limit must be positive")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    model_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Select one authorised GPU by UUID."
    device = torch.device("cuda:0")
    workdir = Path(args.workdir).resolve()
    repo = Path(args.repo).resolve() if args.repo else workdir / "code/matryoshka-attribution"
    root = workdir / "results"
    root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="feedback_mattr_", dir=str(root)))
    logger.info("OUTPUT DIRECTORY: %s", out)
    dump_json(out / "arguments.json", vars(args))
    (out / "repo_commit.txt").write_text(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True))
    factory = official_factory(repo / "scripts/mib/eval_mib_edge.py", out)
    baseline_dir = Path(args.baseline_dir).resolve()
    old_summary = json.loads((baseline_dir / "summary.json").read_text())
    assert old_summary["model"] == MODEL and old_summary["resolved_revision"] == REVISION
    snapshot = Path(old_summary["snapshot"])
    assert snapshot.is_dir(), "Pinned local model snapshot is missing."
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    labels = {s: tokenizer.encode(s, add_special_tokens=False) for s in ["A", "B"]}
    assert all(len(x) == 1 for x in labels.values())
    labels = {s: x[0] for s, x in labels.items()}
    assert labels == old_summary["label_ids"]
    # Auto-detect data version and name from baseline or use explicit overrides
    baseline_version = old_summary.get("version", "v1")
    data_version = args.data_version if args.data_version else baseline_version
    baseline_data_name = old_summary.get("data_name", "feedback_retry_answer")
    data_name = args.data_name if args.data_name else baseline_data_name
    # Construct data path: data/{name}_{version} or data/{name}_v1 if no version suffix
    if data_version:
        data_root = workdir / f"data/{data_name}_{data_version}"
    else:
        data_root = workdir / f"data/{data_name}_v1"
    train = read_pairs(data_root / "train.jsonl", tokenizer, labels, device)
    development = read_pairs(data_root / "dev.jsonl", tokenizer, labels, device)
    saved_tokens = {r["pair_id"]: r for r in map(json.loads, (baseline_dir / "tokenized_pairs.jsonl").read_text().splitlines())}
    # Optional count validation for known datasets
    if not args.skip_count_check:
        expected_counts = {"v1": (64, 32), "v2": (256, 64)}
        if data_name == "feedback_retry_answer" and data_version in expected_counts:
            expected_train, expected_dev = expected_counts[data_version]
            assert len(train) == expected_train and len(development) == expected_dev, \
                f"Expected {expected_train} train / {expected_dev} dev pairs for {data_name} {data_version}, got {len(train)} / {len(development)}"
        elif data_name == "export_goal_verification" and data_version == "v1":
            assert len(train) == 256 and len(development) == 64, \
                f"Expected 256 train / 64 dev pairs for export_goal_verification v1, got {len(train)} / {len(development)}"
        else:
            logger.warning("No count validation for %s %s; loaded %d train / %d dev pairs",
                         data_name, data_version, len(train), len(development))
    for example in development:
        old = saved_tokens[example["pair"]["pair_id"]]
        for side in ["base", "source"]:
            assert example["tokens"][side][0].tolist() == old[side]["input_ids"], "Prompt/token mismatch with baseline."
    evaluate_examples = development[:args.eval_limit]
    longest_train = max(range(len(train)), key=lambda i: train[i]["pretok"][2])
    parity_examples = development + [train[longest_train]]
    logger.info("Comparison precision: %s; full FP32 matmul precision requested.", args.dtype)
    logger.info("Tokenised %d train / %d dev pairs; evaluating first %d dev pairs, no filtering.",
                len(train), len(development), len(evaluate_examples))
    logger.info("Loading pinned HF model for conversion and numerical comparison.")
    hf = AutoModelForCausalLM.from_pretrained(str(snapshot), dtype=model_dtype,
            attn_implementation="eager", local_files_only=True).to(device).eval()
    hf.requires_grad_(False)
    hf_margins = [(e, side, raw_margin(hf, e, side, hf=True))
                  for e in parity_examples for side in ["base", "source"]]
    from matryoshka_attribution.deps import add_mib_to_sys_path
    add_mib_to_sys_path()
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from matryoshka_attribution import learn_scores
    logger.info("Moving HF model to CPU before conversion to avoid two GPU-resident copies.")
    hf.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Converting the SAME checkpoint to HookedTransformer; no weight folding/centering.")
    model = HookedTransformer.from_pretrained(MODEL, hf_model=hf, tokenizer=tokenizer,
        revision=REVISION, device="cuda", dtype=model_dtype,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
        fold_value_biases=False, refactor_factored_attn_matrices=False)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    model.requires_grad_(False)
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    parity = []
    for example, side, reference in hf_margins:
        actual = raw_margin(model, example, side)
        parity.append({"pair_id": example["pair"]["pair_id"], "side": side,
            "HF_margin": reference, "TL_margin": actual,
            "absolute_error": abs(reference-actual),
            "decision_disagrees": reference * actual < 0,
            "either_margin_is_zero": reference == 0 or actual == 0})
    csv_write(out / "hf_tl_comparison.csv", parity)
    max_error = max(r["absolute_error"] for r in parity)
    disagreements = sum(r["decision_disagrees"] for r in parity)
    logger.info("HF/TL comparison: max margin difference %.6f, %d/%d strict non-tie preference reversals.",
                max_error, disagreements, len(parity))
    if max_error > args.max_margin_error or disagreements > args.max_decision_disagreements:
        raise RuntimeError("HF/TL comparison failed the explicit pilot tolerances. Scores have NOT been trained.")
    # Record a new baseline at the requested numerical precision.
    dev_ids = {e["pair"]["pair_id"] for e in development}
    dev_comparisons = [r for r in parity if r["pair_id"] in dev_ids]
    def accuracy_report(field):
        decisions = {}
        by_action = {}  # Generic: works with any action names
        lookup = {e["pair"]["pair_id"]: e["pair"] for e in development}
        for r in dev_comparisons:
            correct = r[field] > 0 if r["side"] == "base" else r[field] < 0
            decisions.setdefault(r["pair_id"], []).append(correct)
            action = lookup[r["pair_id"]][r["side"] + "_action"]
            by_action.setdefault(action, []).append(correct)
        return {"contexts": len(dev_comparisons),
                "pairwise_action_accuracy": sum(sum(v) for v in decisions.values()) / len(dev_comparisons),
                "both_sides_correct_fraction": sum(all(v) for v in decisions.values()) / len(decisions),
                "per_action_accuracy": {k: sum(v)/len(v) for k,v in by_action.items()}}
    precision_baseline = {"dtype": args.dtype, "model": MODEL, "revision": REVISION,
        "HF": accuracy_report("HF_margin"), "TransformerLens": accuracy_report("TL_margin"),
        "max_margin_error": max_error, "strict_preference_reversals": disagreements,
        "comparison_contexts": len(parity),
        "note": "New execution baseline; do not substitute the earlier BF16 accuracy."}
    dump_json(out / "precision_baseline.json", precision_baseline)
    print("\nPRECISION COMPARISON SUMMARY\n" + json.dumps(precision_baseline, indent=2), flush=True)
    if args.conversion_only:
        logger.info("CONVERSION CHECK PASSED. No interventions or training were run. Results: %s", out)
        return
    graph = Graph.from_model(model)
    total = int(graph.real_edge_mask.sum())
    logger.info("Graph: %d scored edges; all model parameters frozen.", total)
    train_env = Environment(factory, model, graph, train, args)
    eval_env = Environment(factory, model, graph, evaluate_examples, args)
    zeros = torch.zeros(total, device=device)
    ones = torch.ones(total, device=device)
    endpoints = []
    # Include the longest training prompt, not only a random short example.
    for i in sorted({0, longest_train}):
        e = train[i]
        full = raw_margin(model, e, "base")
        source_margin = raw_margin(model, e, "source")
        kept = train_env.margin(ones, i)
        patched = train_env.margin(zeros, i)
        endpoints.append({"pair_id": e["pair"]["pair_id"], "base": full,
            "all_kept": kept, "source": source_margin, "all_patched": patched,
            "keep_error": abs(full-kept), "patch_error": abs(source_margin-patched)})
    csv_write(out / "endpoint_checks.csv", endpoints)
    if max(max(r["keep_error"], r["patch_error"]) for r in endpoints) > args.max_margin_error:
        raise RuntimeError("Intervention endpoint checks failed; no training performed.")
    soft = torch.full((total,), 0.5, device=device, requires_grad=True)
    preflight_loss = train_env.fixed_loss(soft, longest_train)
    preflight_loss.backward()
    assert soft.grad is not None and torch.isfinite(soft.grad).all() and soft.grad.abs().max() > 0
    assert all(p.grad is None for p in model.parameters())
    gradient_norm = soft.grad.norm().item()
    del soft, preflight_loss
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("PREFLIGHT PASSED: endpoints checked; gradient norm %.6g; peak allocated %.2f GiB.",
                gradient_norm, torch.cuda.max_memory_allocated()/2**30)
    dump_json(out / "preflight.json", {"model": MODEL, "revision": REVISION,
        "official_source_sha256": EXPECTED_SOURCE_SHA, "train_pairs": len(train),
        "dev_pairs_used_for_curve": len(evaluate_examples), "parity_max_margin_error": max_error,
        "parity_disagreements": disagreements, "gradient_norm": gradient_norm,
        "warning": "Explicit numerical tolerances, not a proof of exact numerical equivalence."})
    if args.preflight_only:
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info("TRAINING: %d steps, official learn_scores, sigmoid top-k, uniform k, Adam.", args.steps)
    result = learn_scores(total, train_env.loss, steps=args.steps, variant="topk",
        k_schedule="uniform", T=args.T, n_iters=args.n_iters, lr=args.lr,
        optimizer="adam", l0_lambda=0.0, device="cuda", logger=logger, log_every=10)
    scores = result.scores.detach().cpu()
    # Save BEFORE the potentially longer development evaluation.
    torch.save({"scores": scores, "args": vars(args), "model": MODEL, "revision": REVISION,
        "loss_log": result.loss_log, "train_time_s": result.train_time_s,
        "graph_cfg": dict(graph.cfg)}, out / "scores.pt")
    logger.info("Scores saved; now evaluating development circuits.")
    refs = []
    for e in evaluate_examples:
        refs.append({"pair_id": e["pair"]["pair_id"], "base": raw_margin(model, e, "base"),
                     "source": raw_margin(model, e, "source")})
    csv_write(out / "full_model_reference.csv", refs)
    graph.scores[:] = -torch.inf
    graph.scores[graph.real_edge_mask.bool()] = scores
    fractions = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.]
    curves, margins = [], []
    for fraction in fractions:
        k = int(fraction * total)
        graph.apply_topn(k, absolute=False, level="edge", prune=True)
        keep = graph.in_graph[graph.real_edge_mask.bool()].float().to(device)
        current = [eval_env.margin(keep, i) for i in range(len(evaluate_examples))]
        faith = [(m-r["source"])/(r["base"]-r["source"])
                 for m,r in zip(current, refs) if abs(r["base"]-r["source"]) > 1e-6]
        agreement = [(m > 0) == (r["base"] > 0)
                     for m,r in zip(current, refs) if r["base"] != 0]
        row = {"fraction": fraction, "nominal_k": k,
            "pruned_edges": int(graph.weighted_edge_count()),
            "IIA": sum(m > 0 for m in current)/len(current),
            "full_model_base_IIA": sum(r["base"] > 0 for r in refs)/len(refs),
            "agreement_with_full_non_ties": sum(agreement)/len(agreement) if agreement else None,
            "faithfulness_nonzero_denominators": sum(faith)/len(faith) if faith else None,
            "faithfulness_defined_examples": len(faith), "evaluation_pairs": len(refs)}
        curves.append(row)
        margins.extend({"fraction": fraction, "pair_id": r["pair_id"], "margin": m}
                       for m,r in zip(current, refs))
        logger.info("%.1f%%: top-%d -> %d edges; IIA %.3f (full base IIA %.3f)",
            100*fraction, k, row["pruned_edges"], row["IIA"], row["full_model_base_IIA"])
        # Save partial evaluation progress after each circuit size.
        csv_write(out / "curve.csv", curves)
        csv_write(out / "circuit_margins.csv", margins)
    compactness = sum(math.log(fractions[i+1]/fractions[i]) *
        (curves[i]["IIA"]+curves[i+1]["IIA"])/2 for i in range(len(fractions)-1)) / math.log(fractions[-1]/fractions[0])
    mapping = {tuple(int(v) for v in edge.matrix_index): edge for edge in graph.edges.values()}
    coords = graph.real_edge_mask.bool().nonzero().tolist()
    top = torch.argsort(scores, descending=True)[:20].tolist()
    named = [{"rank": rank, "score_index": index,
        "edge": mapping[tuple(coords[index])].name, "score": scores[index].item()}
        for rank,index in enumerate(top,1)]
    csv_write(out / "top20_named_edges.csv", named)
    summary = {"model": MODEL, "revision": REVISION, "dtype": args.dtype, "training_steps": args.steps,
        "train_pairs": len(train), "dev_pairs_evaluated": len(refs),
        "total_scored_edges": total, "train_time_s": result.train_time_s,
        "compactness": compactness, "full_model_base_IIA": curves[-1]["full_model_base_IIA"],
        "full_model_source_action_accuracy": sum(r["source"] < 0 for r in refs)/len(refs),
        "notes": "No test data loaded; no correctness filtering. Curve IIA is base-action accuracy on dev pairs, not the 64-context baseline. Near-zero faithfulness denominators explicitly excluded and counts reported."}
    dump_json(out / "summary.json", summary)
    print("\nCUSTOM MATTR SUMMARY\n"+json.dumps(summary, indent=2))
    print("\nSaved custom-task run:", out)


if __name__ == "__main__":
    main()
