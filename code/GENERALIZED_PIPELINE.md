# Generalized Pipeline Guide

This document explains how the feedback circuit extraction pipeline now supports arbitrary workload versions without hard-coded assumptions.

## Overview

All pipeline scripts are now fully generalized to work with any workload version (v1, v2, v3, ...) without modification. The only version-specific files are the workload generators themselves.

## Generalized Scripts

### ✅ Fully Generalized (work with any workload)

| Script | Auto-detection | Notes |
|--------|---------------|-------|
| `run_feedback_mattr_fp32.py` | Reads `version` from baseline `summary.json` | Adapter auto-detects train/dev pair counts |
| `run_feedback_mattr_lowmem.py` | N/A | Launcher has no workload assumptions |
| `eval_feedback_mattr_saved.py` | N/A | Works with any dev set size |
| `eval_feedback_mattr_reverse.py` | N/A | Works with any dev set size |
| `eval_feedback_mattr_test_generalized.py` | Traces circuit → source → baseline to find version | **Fully generalized test evaluator** |

### Version-Specific Files (one per workload)

| File | Purpose |
|------|---------|
| `feedback_pilot.py` | v1 workload generator |
| `feedback_pilot_v2.py` | v2 workload generator |
| `feedback_pilot_v3.py` | (Future) v3 workload generator |

## Creating a New Workload (v3 example)

### Step 1: Create Generator

Copy and modify an existing generator:

```bash
cp code/feedback_pilot_v2.py code/feedback_pilot_v3.py
```

Edit `feedback_pilot_v3.py`:
- Update `COUNTS = {"train": X, "dev": Y, "test": Z}` with your sizes
- Modify `build_pairs()` logic for your changes
- Ensure `version: "v3"` is written to baseline `summary.json`

**Critical:** Include these in each pair's metadata for debugging:
```python
rows.append({
    "pair_id": f"{split}-{i:03d}",
    "split": split,
    # ... your workload fields ...
    "template_index": template_index,  # If templates vary
    "ab_mapping": ab_mapping_choice,   # If mapping varies
    # ... any other factors that could confound
})
```

### Step 2: Generate Data

```bash
python code/feedback_pilot_v3.py --workdir /path/to/workdir --build-only
```

Creates: `data/feedback_retry_answer_v3/{train,dev,test}.jsonl`

### Step 3: Run Baseline

```bash
python code/feedback_pilot_v3.py \
  --workdir /path/to/workdir \
  --model Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --split dev --device cuda
```

Baseline automatically includes `"version": "v3"` in `summary.json`.

### Step 4: Extract Circuit

```bash
python code/run_feedback_mattr_lowmem.py \
  --workdir /path/to/workdir \
  --baseline-dir /path/to/v3_baseline \
  --dtype float32 \
  --steps 50 --lr 0.05 --seed 42 --eval-limit 16
```

Adapter reads `version` from baseline → auto-detects v3 → loads correct data paths.

### Step 5: Evaluate Development

```bash
python code/eval_feedback_mattr_saved.py \
  --workdir /path/to/workdir \
  --scores-run /path/to/v3_extraction
```

Works automatically with any dev set size.

### Step 6: Freeze Circuit

```bash
python code/eval_feedback_mattr_reverse.py \
  --workdir /path/to/workdir \
  --scores-run /path/to/v3_extraction \
  --reference-eval-run /path/to/v3_dev_eval \
  --fraction 0.1
```

Saves `fixed_mask.pt` with metadata.

### Step 7: Held-Out Test

```bash
python code/eval_feedback_mattr_test_generalized.py \
  --workdir /path/to/workdir \
  --circuit-run /path/to/v3_reverse_eval
```

**No `--data-version` needed!** Auto-detects v3 by tracing:
1. Circuit run → source run → baseline
2. Reads `"version": "v3"` from baseline
3. Loads `feedback_pilot_v3.py`
4. Infers test pair count from actual test data
5. Runs evaluation

Optional explicit override:
```bash
python code/eval_feedback_mattr_test_generalized.py \
  --workdir /path/to/workdir \
  --circuit-run /path/to/v3_reverse_eval \
  --data-version v3  # Only if auto-detection fails
```

## How Auto-Detection Works

### In `run_feedback_mattr_fp32.py`

```python
# Reads baseline summary.json
baseline_version = old_summary.get("version", "v1")
data_version = args.data_version if args.data_version else baseline_version
data_root = workdir / f"data/feedback_retry_answer_{data_version}"

# Flexible pair counts
expected_counts = {"v1": (64, 32), "v2": (256, 64), "v3": (512, 128)}
expected_train, expected_dev = expected_counts.get(data_version, (len(train), len(development)))
```

### In `eval_feedback_mattr_test_generalized.py`

```python
# Trace back through metadata
source_run = Path(prior["source_run"]).resolve()
source_args = json.loads((source_run / "arguments.json").read_text())
baseline = json.loads((Path(source_args["baseline_dir"]) / "summary.json").read_text())
data_version = baseline.get("version", "v1")

# Dynamic generator selection
generator_file = "feedback_pilot.py" if data_version == "v1" else f"feedback_pilot_{data_version}.py"
generator = import_file(code / generator_file, "feedback_generator_readonly")

# Infer pair count from actual data
expected_pairs = len(records)  # No hard-coded counts!
```

## No Hard-Coded Assumptions

**Removed from all scripts:**
- ❌ `feedback_retry_answer_v1` paths
- ❌ `len(train) == 64`, `len(development) == 32`
- ❌ Hard-coded edge counts (91,354 or 91,932)
- ❌ Hard-coded mask hashes
- ❌ Version enumerations in argparse choices

**Added flexibility:**
- ✅ Version read from baseline metadata
- ✅ Pair counts inferred from actual data
- ✅ Edge counts from saved mask
- ✅ Generator file dynamically selected

## Workload Design Best Practices

### Avoid Spurious Correlations

**Bad (v1 mistake):**
```python
# Template and mapping both use i % 2 → confounded!
template_index = i % len(TEMPLATES[split])
retry_code, answer_code = ("A", "B") if i % 2 == 0 else ("B", "A")
```

**Good (v2 fix):**
```python
# Independent randomization
template_index = i % len(TEMPLATES[split])
ab_mapping_choice = rng.randint(0, 1)  # Independent!
retry_code, answer_code = ("A", "B") if ab_mapping_choice == 0 else ("B", "A")
```

### Full Factorial Design

Ensure all factor combinations appear in training:
- 2 templates × 2 mappings × 2 actions = 8 combinations
- Not just 2 or 4

### Test with Unseen Surface Features

Change templates/wording in test split to verify circuit learned mechanism, not shortcuts.

### Include Factor Metadata

Always save `template_index`, `ab_mapping`, etc. for post-hoc diagnosis.

## Debugging Circuit Failures

If circuit fails asymmetrically (e.g., RETRY 100%, ANSWER 50%):

1. **Copy paired results:**
```bash
scp server:/path/to/test_run/paired_results.csv local/
```

2. **Analyze by factors:**
```python
import pandas as pd
df = pd.read_csv('paired_results.csv')

# Check if failures correlate with label
print(df.groupby('base_label')['original_correct'].mean())

# Check if failures correlate with template (if metadata available)
test_data = [json.loads(line) for line in open('data/.../test.jsonl')]
meta = pd.DataFrame(test_data)
merged = df.merge(meta, on='pair_id')
print(merged.groupby('template_index')['original_correct'].mean())
```

3. **Look for perfect correlation** = confound detected

## Version History

| Version | Train | Dev | Test | Key Changes |
|---------|-------|-----|------|-------------|
| v1 | 64 | 32 | 32 | Original (template/mapping confound) |
| v2 | 256 | 64 | 64 | Decoupled factors, 4× scale |

## Migration Notes

**Old v1-locked test script:** `eval_feedback_mattr_test.py`
- Hard-coded to 91,354 edges, specific mask hash
- Preserved for v1 historical reproduction only
- **Do not use for new workloads**

**New generalized script:** `eval_feedback_mattr_test_generalized.py`
- Works with any workload version
- No hard-coded edge counts or hashes
- Auto-detects version from metadata
- **Use this for all v2+ workloads**

## Summary

**For any new workload:**
1. Copy `feedback_pilot_v2.py` → `feedback_pilot_vN.py`
2. Modify generator logic, set `version: "vN"` in baseline
3. Run pipeline normally - all scripts auto-detect vN
4. Use `eval_feedback_mattr_test_generalized.py` for test

**Zero pipeline script changes needed!**
