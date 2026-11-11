#!/usr/bin/env python3
"""
Simplified export goal verification workload (v2).

Reduces variation to improve baseline accuracy:
- Only 2 count field names (rows_written, records_exported)
- Only 2 status values (completed, finished)
- Only 4 item types
- No distracting fields
- Still varies templates across splits for generalization
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Literal

# Action labels
FINISH = "FINISH"
REPAIR_EXPORT = "REPAIR_EXPORT"

# Reduced item types (4 instead of 8)
ITEM_TYPES = [
    "user_records",
    "product_catalog",
    "transaction_log",
    "order_history",
]

# Templates per split (still varied for generalization test)
TEMPLATES = {
    "train": [
        "Export exactly {count} {item_type} to CSV.",
        "Write {count} {item_type} to the output file.",
    ],
    "dev": [
        "Save {count} {item_type} in CSV format.",
    ],
    "test": [
        "Extract {count} {item_type} to the data file.",
    ],
}

# Only 2 count field names (reduced from 8)
COUNT_FIELD_NAMES = [
    "rows_written",
    "records_exported",
]

# Only 2 status values (reduced from 4)
STATUS_VALUES = [
    "completed",
    "finished",
]

# No distracting fields in v2 - keep JSON response simple

# Requested counts - all must tokenize as 3 tokens (100+)
COUNTS = [110, 120, 130, 140]

# For mismatches: actual = requested - offset
# Ensure actual stays >= 100 (3 tokens): min is 110-10=100
MISMATCH_OFFSETS = [5, 10, 15, 20]  # Results: 105-135, all 3-token


def build_pairs(split: Literal["train", "dev", "test"], count: int, seed: int) -> list[dict]:
    """
    Build paired observations for simplified export goal verification.

    Reduced variation:
    - 2 count fields (was 8)
    - 2 status values (was 4)
    - 4 item types (was 8)
    - No distracting fields (was 5)
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

        # Count field name: cycle deterministically (only 2 options)
        count_field = COUNT_FIELD_NAMES[i % len(COUNT_FIELD_NAMES)]

        # Status value: cycle deterministically (only 2 options)
        status_value = STATUS_VALUES[i % len(STATUS_VALUES)]

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

        # Build tool responses - simple, no distracting fields
        # Use integers (not padded strings) - all 3-digit numbers tokenize consistently
        base_tool_response = {
            "status": status_value,
            count_field: base_actual,
        }
        source_tool_response = {
            "status": status_value,
            count_field: source_actual,
        }

        base_tool_msg = json.dumps(base_tool_response)
        source_tool_msg = json.dumps(source_tool_response)

        # Build system message with clear action definitions
        system_msg = (
            "You are verifying an export operation. The user requested a specific number of records. "
            "The export tool has returned its result. Compare the actual count with the requested count.\n\n"
            f"If the counts match exactly, select FINISH (the export succeeded).\n"
            f"If the counts don't match, select REPAIR_EXPORT (the export needs to be fixed).\n\n"
            f"{finish_label} means FINISH; {repair_label} means REPAIR_EXPORT.\n"
            f"Respond with exactly one letter. Do not explain."
        )

        instruction = f"The export tool has finished. Select the next-action letter."

        base_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": base_user_msg},
            {"role": "assistant", "content": "<tool_call>\nexport_csv\n</tool_call>"},
            {"role": "user", "content": f"<tool_response>\n{base_tool_msg}\n</tool_response>\n\n{instruction}"},
        ]

        source_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": source_user_msg},
            {"role": "assistant", "content": "<tool_call>\nexport_csv\n</tool_call>"},
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
    """Save train/dev/test splits."""
    data_dir = workdir / "data" / "export_goal_verification_v2"
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


# Generate locally without torch dependency
import sys
workdir = Path("C:/Users/wilso/OneDrive/Documents/MAttr")
overwrite = "--overwrite" in sys.argv
save_workload(workdir, overwrite=overwrite)

print("\nSimplified v2 workload generation complete!")
print("\nSimplifications from v1:")
print("- Count fields: 8 → 2 (rows_written, records_exported)")
print("- Status values: 4 → 2 (completed, finished)")
print("- Item types: 8 → 4")
print("- Distracting fields: 5 patterns → 0 (removed)")
print("- Templates: still varied across splits for generalization")
