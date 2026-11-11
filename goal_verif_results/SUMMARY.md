# Export Goal Verification v2 - MAttr Circuit Extraction Results

**Date:** 2026-11-08  
**Model:** Qwen2.5-7B-Instruct (revision a09a35458c702b33eeacc393d103063234e8bc28)  
**Workload:** export_goal_verification_v2 (64 test pairs)  
**Task:** Verify if export tool output count matches requested count → FINISH (match) or REPAIR_EXPORT (mismatch)

---

## Pipeline Results

### 1. Baseline (Full Model)
- **Development (64 pairs):** 100% accuracy (64/64)
  - FINISH: 100% (32/32)
  - REPAIR_EXPORT: 100% (32/32)
- **Conclusion:** Model can perform the task perfectly

### 2. Extraction (50 steps, FP32, 5% nominal budget)
- **Training:** 64 pairs (reduced from 256 to avoid OOM)
- **Output:** `results/feedback_mattr_5auv39l6/scores.pt`
- **Dev curve (8 pairs):** 5% and 10% budgets both achieved 100% IIA
- **Extraction completed successfully** after fixing:
  - Training set size (256→64 pairs)
  - GPU selection (GPU 2→5 based on memory)
  - Action name generalization

### 3. Development Evaluation (32 pairs, all budgets)
- **5% budget (46,197 edges):** 100% IIA (32/32)
- **10% budget (93,069 edges):** 100% IIA (32/32)
- **Selected:** 5% budget for smaller circuit

### 4. Reverse Evaluation (32 pairs, both directions)
- **Original direction:** 100% (32/32)
- **Reversed direction:** 100% (32/32)
- **Paired both-sides-correct:** 100% (32/32)
- **Frozen mask:** 46,197 edges, SHA256 `18984364e868008d...`

### 5. **Held-Out Test Evaluation (64 pairs, both directions)**

**CRITICAL FAILURE:**

| Metric | Circuit | Full Model |
|--------|---------|------------|
| **FINISH accuracy** | 59.4% (38/64) | 90.6% (58/64) |
| **REPAIR_EXPORT accuracy** | 53.1% (34/64) | 89.1% (57/64) |
| **Combined accuracy** | 56.2% (72/128) | 89.8% (115/128) |
| **Both-sides-correct** | 12.5% (8/64) | 79.7% (51/64) |

**Circuit performs at chance level on held-out test despite 100% development accuracy.**

---

## Root Cause Analysis

### What Went Wrong

1. **Not a label shortcut:** Label A: 52.9%, Label B: 53.3% (both ~chance)
2. **Not an action shortcut:** Both actions ~50-60% (not learning "always FINISH" or "always REPAIR")
3. **Not template confound:** v2 decoupled all factors (template, mapping, action)
4. **Perfect development → random test:** Classic overfitting, but to what?

### Hypotheses

**Hypothesis 1: Small training set (64 pairs)**
- Reduced from 256 pairs to avoid OOM during TL conversion
- May have insufficient diversity for circuit to learn genuine count-matching logic
- Circuit memorized development examples rather than learning the comparison pattern

**Hypothesis 2: Extraction setup mismatch**
- Extraction used 8 dev pairs during training (eval-limit=8)
- Reverse eval used all 32 dev pairs and achieved 100%
- Circuit may have overfit to those specific 8 pairs or their distribution

**Hypothesis 3: Task complexity**
- Count-matching requires:
  1. Parse JSON tool response
  2. Extract count field value
  3. Compare with requested count from prompt
  4. Map match/mismatch to correct action label
- May be too complex for 5% of edges with only 64 training examples

**Hypothesis 4: Insufficient signal in learned scores**
- Sigmoid top-k selected 46,197 edges from learned scores
- Scores may have ranked edges by spurious correlations that worked on dev but not test
- 50 training steps may be insufficient for convergence

### What Worked vs What Failed

✅ **Successful components:**
- Workload generation (100% baseline)
- TL conversion (max error 0.000145)
- Generic action handling
- Factor decoupling (no template/mapping confound)
- Pipeline infrastructure

❌ **Failed components:**
- Generalization to held-out test
- Learning genuine count-matching vs memorization
- Circuit extraction with limited training data

---

## Comparison with Feedback v2 Results

### Feedback v2 (successful)
- **Task:** Check if requested field has numeric value → ANSWER vs RETRY
- **Training:** 256 pairs
- **Test accuracy:** 92.97% combined (100% RETRY, 85.9% ANSWER)
- **Circuit learned genuine feedback-dependent reasoning**

### Export_goal v2 (failed)
- **Task:** Check if count matches → FINISH vs REPAIR_EXPORT
- **Training:** 64 pairs (reduced due to OOM)
- **Test accuracy:** 56.2% combined (~random)
- **Circuit learned nothing generalizable**

**Key difference:** 4× smaller training set likely insufficient for learning comparison logic.

---

## Recommendations

### Immediate Next Steps

1. **Increase training set to 256 pairs:**
   - Use GPU with sufficient memory (GPU 5: 57GB free)
   - Consider gradient checkpointing if still OOM
   - May require overnight run

2. **Verify extraction convergence:**
   - Check if scores.pt shows clear top-edge structure
   - Plot score distribution
   - Verify loss decreased over 50 steps

3. **Diagnostic extraction with different settings:**
   - Try 10% or 20% budget (more edges)
   - Increase training steps to 100
   - Use full dev set (32 pairs) during extraction instead of 8

### Alternative Approaches

1. **Simplify task:**
   - Binary: count > 0 vs count == 0
   - Remove JSON parsing complexity
   - Test if circuit can learn simpler comparison

2. **Increase model capacity:**
   - Use 10% or 20% budget
   - May need more edges to represent comparison logic

3. **Pre-train on auxiliary task:**
   - First train circuit to identify count field
   - Then train to compare values
   - Staged learning approach

---

## Files

### Local Results (`goal_verif_results/`)
- `extraction/` - Extraction run with scores.pt
- `dev_eval/` - Full development evaluation curves
- `reverse_eval/` - Bidirectional evaluation + frozen mask
- `test_eval/` - Held-out test results (FAILED)
- `baseline/` - Full model baseline

### Server Paths
- Workload: `/ssddata/wilson_urop_YeAlWoYJ/data/export_goal_verification_v2/`
- Scripts: `/ssddata/wilson_urop_YeAlWoYJ/code/`
- Results: `/ssddata/wilson_urop_YeAlWoYJ/results/`

---

## Lessons Learned

1. **Training set size matters critically:**
   - Reducing from 256→64 pairs likely caused generalization failure
   - OOM workarounds should not compromise training data quantity

2. **Development accuracy ≠ test accuracy:**
   - 100% dev with 56% test = severe overfitting
   - Need larger held-out validation set during extraction

3. **Task complexity requires sufficient capacity:**
   - Count-matching may need >5% of edges
   - Or more training examples with current budget

4. **Early stopping needed:**
   - Should have caught memorization during extraction
   - Monitor generalization gap continuously

---

## Status

**Pipeline completion:** ✅ All stages executed successfully  
**Circuit quality:** ❌ Failed to learn generalizable behavior  
**Next action:** Retry extraction with 256 training pairs or diagnose score quality
