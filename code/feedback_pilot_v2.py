
import argparse
import csv
import json
from pathlib import Path
import random
import tempfile

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPLATES = {
    "train": ["What is the {field} of {item}?", "Find the {field} for {item}."],
    "dev": ["Please tell me the {field} associated with {item}."],
    "test": ["I need the {field} for {item}; obtain it before answering."],
}
COUNTS = {"train": 256, "dev": 64, "test": 64}
FIELD_PAIRS = [("price", "stock"), ("weight", "length"), ("capacity", "duration")]


def build_pairs(split, count, seed=42):
    rng = random.Random(seed + {"train": 0, "dev": 1, "test": 2}[split])
    rows = []
    for i in range(count):
        fields = list(FIELD_PAIRS[i % len(FIELD_PAIRS)])
        rng.shuffle(fields)
        requested, other = fields
        item = f"{split}-item-{i:03d}"
        value = str(rng.randint(10, 99))
        missing = rng.choice(["unknown", "unavailable"] if split != "test"
                             else ["not supplied", "not recorded"])

        # FIXED: Decouple template selection from A/B mapping
        template_index = i % len(TEMPLATES[split])
        ab_mapping_choice = rng.randint(0, 1)  # Independent random choice
        retry_code, answer_code = ("A", "B") if ab_mapping_choice == 0 else ("B", "A")

        # Action alternates as before (balanced)
        base_action = "retry" if (i // 2) % 2 == 0 else "answer"

        request = TEMPLATES[split][template_index].format(field=requested, item=item)
        system = (
            "You are selecting the next action after a lookup tool result. "
            "If the requested field has a numeric value, ANSWER using that value. "
            "If the requested field is missing or unknown, RETRY the lookup tool. "
            "A numeric value for another field does not answer the request. "
            "Do not perform the action or give the value yet. "
            f"Return exactly one letter: {retry_code} means RETRY; "
            f"{answer_code} means ANSWER. Do not explain."
        )
        contexts = {}
        for action in ["retry", "answer"]:
            record = {requested: missing if action == "retry" else value,
                      other: value if action == "retry" else missing}
            observation = {"status": "ok", "item": item, "record": record}
            user = (
                f"User request: {request}\n"
                f"Previous tool call: lookup(item={item!r})\n"
                "Tool observation:\n" + json.dumps(observation, sort_keys=True) +
                "\nSelect the next-action letter."
            )
            contexts[action] = [{"role": "system", "content": system},
                                {"role": "user", "content": user}]
        source_action = "answer" if base_action == "retry" else "retry"
        codes = {"retry": retry_code, "answer": answer_code}
        rows.append({"pair_id": f"{split}-{i:03d}", "split": split,
                     "requested_field": requested, "other_field": other,
                     "base_action": base_action, "source_action": source_action,
                     "base_label": codes[base_action], "source_label": codes[source_action],
                     "template_index": template_index, "ab_mapping": ab_mapping_choice,
                     "base_messages": contexts[base_action],
                     "source_messages": contexts[source_action]})
    return rows


def save_workload(workdir):
    directory = workdir / "data" / "feedback_retry_answer_v2"
    directory.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for split, count in COUNTS.items():
        rows = build_pairs(split, count)
        all_data[split] = rows
        text = "".join(json.dumps(row) + "\n" for row in rows)
        path = directory / f"{split}.jsonl"
        if path.exists():
            if path.read_text() != text:
                raise RuntimeError(f"Refusing to overwrite different existing data: {path}")
        else:
            with path.open("x") as f:
                f.write(text)
    print("Workload v2:", directory)
    print("Pairs:", COUNTS)

    # Verify factor independence
    train_data = all_data["train"]
    template_action_pairs = set()
    template_mapping_pairs = set()
    action_mapping_pairs = set()
    for row in train_data:
        template_action_pairs.add((row["template_index"], row["base_action"]))
        template_mapping_pairs.add((row["template_index"], row["ab_mapping"]))
        action_mapping_pairs.add((row["base_action"], row["ab_mapping"]))

    print("\nFactor independence check (training):")
    print(f"  Template × Action combinations: {len(template_action_pairs)} (expect 4)")
    print(f"  Template × Mapping combinations: {len(template_mapping_pairs)} (expect 4)")
    print(f"  Action × Mapping combinations: {len(action_mapping_pairs)} (expect 4)")

    action_counts = {"retry": 0, "answer": 0}
    mapping_counts = {0: 0, 1: 0}
    for row in train_data:
        action_counts[row["base_action"]] += 1
        mapping_counts[row["ab_mapping"]] += 1
    print(f"  Action balance: RETRY={action_counts['retry']}, ANSWER={action_counts['answer']}")
    print(f"  Mapping balance: A=RETRY/B=ANSWER={mapping_counts[0]}, B=RETRY/A=ANSWER={mapping_counts[1]}")

    example = all_data["dev"][0]
    print("\nEXAMPLE PAIR (v2)")
    print(json.dumps(example, indent=2))
    return directory, all_data


def baseline(args, directory, data):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA unavailable"
        assert torch.cuda.device_count() == 1, "Select exactly one GPU with CUDA_VISIBLE_DEVICES"
    workdir = Path(args.workdir).resolve()
    result_root = workdir / "results"
    result_root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"feedback_baseline_v2_{args.split}_", dir=str(result_root)))
    snapshot = snapshot_download(args.model, revision=args.revision, allow_patterns=[
        "*.json", "*.safetensors", "*.model", "merges.txt", "*.tiktoken"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    label_ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in ["A", "B"]}
    assert all(len(ids) == 1 for ids in label_ids.values()), f"Labels are not single-token: {label_ids}"
    label_ids = {label: ids[0] for label, ids in label_ids.items()}
    print("Label IDs:", label_ids)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=dtype, attn_implementation="eager", local_files_only=True,
    ).to(args.device).eval()
    model.requires_grad_(False)
    predictions, tokenized_pairs = [], []
    preview = []
    with torch.inference_mode():
        for pair in data[args.split]:
            encoded = {}
            for side in ["base", "source"]:
                text = tokenizer.apply_chat_template(pair[f"{side}_messages"],
                    tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
                ids = inputs["input_ids"][0].tolist()
                encoded[side] = {"prompt": text, "input_ids": ids}
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                logits = model(**inputs, use_cache=False).logits[0, -1].float()
                label = pair[f"{side}_label"]
                alternative = "B" if label == "A" else "A"
                margin = (logits[label_ids[label]] - logits[label_ids[alternative]]).item()
                argmax = int(logits.argmax())
                predictions.append({"pair_id": pair["pair_id"], "side": side,
                    "action": pair[f"{side}_action"], "expected_label": label,
                    "positive_margin": margin > 0, "margin": margin,
                    "next_token_correct": argmax == label_ids[label],
                    "next_token": tokenizer.decode([argmax]), "token_length": len(ids)})
                if len(preview) < 4:
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                                               pad_token_id=tokenizer.eos_token_id)
                    completion = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
                    preview.append({"pair_id": pair["pair_id"], "side": side,
                                    "expected": label, "generated": completion})
            tokenized_pairs.append({"pair_id": pair["pair_id"],
                "equal_token_length": len(encoded["base"]["input_ids"]) == len(encoded["source"]["input_ids"]),
                "base_label_id": label_ids[pair["base_label"]],
                "source_label_id": label_ids[pair["source_label"]], **encoded})
    n = len(predictions)
    pair_both = sum(predictions[2*i]["positive_margin"] and predictions[2*i+1]["positive_margin"]
                    for i in range(len(data[args.split])))
    equal = sum(p["equal_token_length"] for p in tokenized_pairs)
    summary = {"model": args.model, "snapshot": snapshot,
        "resolved_revision": Path(snapshot).name, "split": args.split, "version": "v2",
        "pairs": len(data[args.split]), "contexts": n,
        "pairwise_action_accuracy": sum(r["positive_margin"] for r in predictions)/n,
        "both_sides_correct_fraction": pair_both/len(data[args.split]),
        "full_vocabulary_next_token_accuracy": sum(r["next_token_correct"] for r in predictions)/n,
        "equal_length_pairs": equal, "label_ids": label_ids,
        "per_action_accuracy": {action: sum(r["positive_margin"] for r in predictions if r["action"] == action)
            / sum(r["action"] == action for r in predictions) for action in ["retry", "answer"]},
        "notes": "v2 workload: template selection decoupled from A/B mapping. Constrained next-action selection from simulated feedback, not native tool calling. No examples filtered by correctness or length.",
        "generation_preview": preview}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    (out / "tokenized_pairs.jsonl").write_text("".join(json.dumps(p)+"\n" for p in tokenized_pairs))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("\nBASELINE SUMMARY (v2)\n" + json.dumps(summary, indent=2))
    print("\nSaved baseline:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    directory, data = save_workload(Path(args.workdir).resolve())
    if not args.build_only:
        baseline(args, directory, data)


if __name__ == "__main__":
    main()
