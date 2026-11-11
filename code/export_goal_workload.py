#!/usr/bin/env python3
"""
Export goal verification workload generator.

Tests whether circuit checks actual outcome vs requested goal.
Tool always returns execution=completed; model must compare rows_written with requested count.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Literal
import hashlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Action labels
FINISH = "FINISH"
REPAIR_EXPORT = "REPAIR_EXPORT"

# Item types to export
ITEM_TYPES = [
    "user_records",
    "product_catalog",
    "transaction_log",
    "order_history",
    "customer_profiles",
    "inventory_items",
    "sales_reports",
    "audit_entries",
]

# Templates per split (deterministic selection, now include item types)
TEMPLATES = {
    "train": [
        "Export exactly {count} {item_type} to CSV.",
        "Write {count} {item_type} to the output file.",
        "Save {count} {item_type} in CSV format.",
        "Extract {count} {item_type} to the data file.",
    ],
    "dev": [
        "Generate a CSV with {count} {item_type}.",
        "Create an export of {count} {item_type}.",
    ],
    "test": [
        "Produce a CSV file containing {count} {item_type}.",
        "Output {count} {item_type} to the export file.",
    ],
}

# JSON response field names for the count
COUNT_FIELD_NAMES = [
    "rows_written",
    "records_exported",
    "entries_saved",
    "items_written",
    "count",
    "total_records",
    "rows_exported",
    "entries_processed",
]

# Status field values
STATUS_VALUES = [
    "completed",
    "finished",
    "done",
    "success",
]

# Distracting fields to add to tool responses
DISTRACTING_FIELDS = [
    lambda i: {"timestamp": f"2026-11-07T13:{50+i%10:02d}:00Z"},
    lambda i: {"file_path": f"/exports/data_{i:04d}.csv"},
    lambda i: {"duration_ms": 100 + i * 13},
    lambda i: {"checksum": f"md5:{hash(i) & 0xFFFFFFFF:08x}"},
    lambda i: {},  # Sometimes no extra field
]

# Requested counts
COUNTS = [50, 100, 150, 200]

# For mismatches: actual = requested - offset
MISMATCH_OFFSETS = [10, 20, 30, 40]


def build_pairs(split: Literal["train", "dev", "test"], count: int, seed: int) -> list[dict]:
    """
    Build paired observations for export goal verification.

    Each pair:
      Base: one action (FINISH or REPAIR_EXPORT)
      Source: opposite action

    Factors decoupled:
      - Template: deterministic from split templates
      - Item type: deterministic cycle
      - Count field name: deterministic cycle
      - Status value: deterministic cycle
      - Distracting fields: deterministic per index
      - A/B mapping: randomized independently
      - Action: depends on goal match
    """
    rng = random.Random(seed)
    templates = TEMPLATES[split]
    pairs = []

    for i in range(count):
        # Template: cycle deterministically
        template = templates[i % len(templates)]

        # Item type: cycle deterministically
        item_type = ITEM_TYPES[i % len(ITEM_TYPES)]

        # Requested count: cycle through COUNTS
        requested = COUNTS[i % len(COUNTS)]

        # Count field name: cycle deterministically
        count_field = COUNT_FIELD_NAMES[i % len(COUNT_FIELD_NAMES)]

        # Status value: cycle deterministically
        status_value = STATUS_VALUES[i % len(STATUS_VALUES)]

        # Distracting fields: deterministic per index
        distracting = DISTRACTING_FIELDS[i % len(DISTRACTING_FIELDS)](i)

        # A/B mapping: random
        ab_mapping = rng.randint(0, 1)
        if ab_mapping == 0:
            finish_label = "A"
            repair_label = "B"
        else:
            finish_label = "B"
            repair_label = "A"

        # One context matches goal (FINISH), other doesn't (REPAIR_EXPORT)
        actual_match = requested
        mismatch_offset = MISMATCH_OFFSETS[i % len(MISMATCH_OFFSETS)]
        actual_mismatch = requested - mismatch_offset

        # Decide which is base (alternating for balance)
        if i % 2 == 0:
            base_action = FINISH
            base_actual = actual_match
            base_label = finish_label
            source_action = REPAIR_EXPORT
            source_actual = actual_mismatch
            source_label = repair_label
        else:
            base_action = REPAIR_EXPORT
            base_actual = actual_mismatch
            base_label = repair_label
            source_action = FINISH
            source_actual = actual_match
            source_label = finish_label

        # Build messages
        base_user_msg = template.format(count=requested, item_type=item_type)
        source_user_msg = base_user_msg  # Same request

        # Build tool responses with varied structure
        base_tool_response = {
            "status": status_value,
            count_field: base_actual,
            **distracting
        }
        source_tool_response = {
            "status": status_value,
            count_field: source_actual,
            **distracting
        }

        base_tool_msg = json.dumps(base_tool_response)
        source_tool_msg = json.dumps(source_tool_response)

        instruction = "The export tool has finished. Respond with exactly one letter: your next action."

        base_messages = [
            {"role": "user", "content": base_user_msg},
            {"role": "assistant", "content": f"<tool_call>\nexport_csv\n</tool_call>"},
            {"role": "user", "content": f"<tool_response>\n{base_tool_msg}\n</tool_response>\n\n{instruction}"},
        ]

        source_messages = [
            {"role": "user", "content": source_user_msg},
            {"role": "assistant", "content": f"<tool_call>\nexport_csv\n</tool_call>"},
            {"role": "user", "content": f"<tool_response>\n{source_tool_msg}\n</tool_response>\n\n{instruction}"},
        ]

        pairs.append({
            "pair_id": f"{split}_{i:03d}",
            "base_messages": base_messages,
            "base_label": base_label,
            "base_action": base_action,
            "base_requested": requested,
            "base_actual": base_actual,
            "source_messages": source_messages,
            "source_label": source_label,
            "source_action": source_action,
            "source_requested": requested,
            "source_actual": source_actual,
            "metadata": {
                "template_index": i % len(templates),
                "item_type": item_type,
                "count_field": count_field,
                "status_value": status_value,
                "ab_mapping": ab_mapping,
                "requested_count": requested,
            }
        })

    return pairs


def save_workload(workdir: Path, overwrite: bool = False):
    """Save train/dev/test splits. Refuse to overwrite differing content."""
    data_dir = workdir / "data" / "export_goal_verification_v1"
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (256, 42),
        "dev": (64, 43),
        "test": (64, 44),
    }

    for split, (count, seed) in splits.items():
        path = data_dir / f"{split}.jsonl"
        pairs = build_pairs(split, count, seed)

        new_content = "".join(json.dumps(p) + "\n" for p in pairs)
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()

        if path.exists():
            existing_content = path.read_text()
            existing_hash = hashlib.sha256(existing_content.encode()).hexdigest()

            if existing_hash == new_hash:
                print(f"[{split}] Unchanged: {path}")
                continue

            if not overwrite:
                raise FileExistsError(
                    f"{path} exists with different content.\n"
                    f"Existing SHA-256: {existing_hash}\n"
                    f"New SHA-256: {new_hash}\n"
                    "Pass --overwrite to replace."
                )

        path.write_text(new_content)
        print(f"[{split}] Wrote {len(pairs)} pairs to {path} (SHA-256: {new_hash[:16]}...)")


def baseline(
    workdir: Path,
    split: Literal["train", "dev", "test"],
    model_name: str,
    revision: str | None,
    device: str,
):
    """
    Evaluate full model on one split.
    Reports A-vs-B pairwise preference accuracy and per-action accuracy.
    """
    data_dir = workdir / "data" / "export_goal_verification_v1"
    path = data_dir / f"{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Run with --build-only first to create {path}")

    pairs = [json.loads(line) for line in path.read_text().splitlines()]

    # Load model
    print(f"Loading {model_name} (revision={revision}) on {device}...")
    load_kwargs = {"device_map": device}
    if revision:
        load_kwargs["revision"] = revision

    # Precision: CUDA uses BF16, CPU uses FP32
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.bfloat16
        dtype_str = "bfloat16"
    else:
        load_kwargs["torch_dtype"] = torch.float32
        dtype_str = "float32"

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    print(f"Model loaded in {dtype_str}.")

    # Token IDs for A and B
    id_A = tokenizer.encode("A", add_special_tokens=False)[0]
    id_B = tokenizer.encode("B", add_special_tokens=False)[0]

    results = []
    correct_pairwise = 0
    action_correct = {FINISH: 0, REPAIR_EXPORT: 0}
    action_total = {FINISH: 0, REPAIR_EXPORT: 0}

    for pair in pairs:
        # Tokenize
        base_text = tokenizer.apply_chat_template(pair["base_messages"], tokenize=False, add_generation_prompt=True)
        source_text = tokenizer.apply_chat_template(pair["source_messages"], tokenize=False, add_generation_prompt=True)

        base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
        source_ids = tokenizer.encode(source_text, return_tensors="pt").to(device)

        # Forward
        with torch.no_grad():
            base_out = model(base_ids)
            source_out = model(source_ids)

        base_logits = base_out.logits[0, -1]
        source_logits = source_out.logits[0, -1]

        base_logit_A = base_logits[id_A].item()
        base_logit_B = base_logits[id_B].item()
        source_logit_A = source_logits[id_A].item()
        source_logit_B = source_logits[id_B].item()

        # Predicted labels
        base_pred = "A" if base_logit_A > base_logit_B else "B"
        source_pred = "A" if source_logit_A > source_logit_B else "B"

        # Pairwise correct: base prefers base_label over source_label
        base_label = pair["base_label"]
        source_label = pair["source_label"]

        if base_label == "A":
            base_prefers_self = base_logit_A > base_logit_B
        else:
            base_prefers_self = base_logit_B > base_logit_A

        pairwise_correct = base_prefers_self
        if pairwise_correct:
            correct_pairwise += 1

        # Action accuracy
        base_action = pair["base_action"]
        source_action = pair["source_action"]

        action_total[base_action] += 1
        action_total[source_action] += 1

        if base_pred == base_label:
            action_correct[base_action] += 1
        if source_pred == source_label:
            action_correct[source_action] += 1

        results.append({
            "pair_id": pair["pair_id"],
            "base_label": base_label,
            "base_pred": base_pred,
            "base_action": base_action,
            "base_requested": pair["base_requested"],
            "base_actual": pair["base_actual"],
            "source_label": source_label,
            "source_pred": source_pred,
            "source_action": source_action,
            "source_requested": pair["source_requested"],
            "source_actual": pair["source_actual"],
            "pairwise_correct": pairwise_correct,
        })

    # Summary
    pairwise_acc = correct_pairwise / len(pairs)
    finish_acc = action_correct[FINISH] / action_total[FINISH] if action_total[FINISH] > 0 else 0
    repair_acc = action_correct[REPAIR_EXPORT] / action_total[REPAIR_EXPORT] if action_total[REPAIR_EXPORT] > 0 else 0

    summary = {
        "model": model_name,
        "revision": revision,
        "resolved_revision": revision,  # For compatibility with adapter expectations
        "device": device,
        "dtype": dtype_str,
        "split": split,
        "version": "v1",
        "data_name": "export_goal_verification",
        "total_pairs": len(pairs),
        "pairwise_accuracy": pairwise_acc,
        "label_ids": {"A": id_A, "B": id_B},
        "snapshot": str(Path(tokenizer.name_or_path).resolve()) if hasattr(tokenizer, 'name_or_path') else model_name,
        "action_accuracy": {
            FINISH: {
                "correct": action_correct[FINISH],
                "total": action_total[FINISH],
                "accuracy": finish_acc,
            },
            REPAIR_EXPORT: {
                "correct": action_correct[REPAIR_EXPORT],
                "total": action_total[REPAIR_EXPORT],
                "accuracy": repair_acc,
            },
        },
    }

    # Save
    run_name = f"baseline_{split}_{hashlib.sha256(str(summary).encode()).hexdigest()[:8]}"
    out_dir = workdir / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "predictions.csv").write_text(
        "pair_id,base_label,base_pred,base_action,source_label,source_pred,source_action,pairwise_correct\n"
        + "\n".join(
            f"{r['pair_id']},{r['base_label']},{r['base_pred']},{r['base_action']},"
            f"{r['source_label']},{r['source_pred']},{r['source_action']},{r['pairwise_correct']}"
            for r in results
        )
    )

    # Save tokenized pairs for adapter conversion check
    tokenized_pairs = []
    for pair in pairs:
        base_text = tokenizer.apply_chat_template(pair["base_messages"], tokenize=False, add_generation_prompt=True)
        source_text = tokenizer.apply_chat_template(pair["source_messages"], tokenize=False, add_generation_prompt=True)
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)
        source_ids = tokenizer.encode(source_text, add_special_tokens=False)

        tokenized_pairs.append({
            "pair_id": pair["pair_id"],
            "base": {"input_ids": base_ids},
            "source": {"input_ids": source_ids},
        })

    (out_dir / "tokenized_pairs.jsonl").write_text(
        "\n".join(json.dumps(tp) for tp in tokenized_pairs) + "\n"
    )

    print(f"\n[Baseline {split}]")
    print(f"Pairwise accuracy: {pairwise_acc:.2%} ({correct_pairwise}/{len(pairs)})")
    print(f"{FINISH}: {finish_acc:.2%} ({action_correct[FINISH]}/{action_total[FINISH]})")
    print(f"{REPAIR_EXPORT}: {repair_acc:.2%} ({action_correct[REPAIR_EXPORT]}/{action_total[REPAIR_EXPORT]})")
    print(f"Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export goal verification workload generator")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true", help="Only build JSONL, skip baseline")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing splits with different content")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="Split for baseline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    save_workload(args.workdir, overwrite=args.overwrite)

    if not args.build_only:
        baseline(args.workdir, args.split, args.model, args.revision, args.device)


if __name__ == "__main__":
    main()
