# V-JEPA 2 Feature Extraction — Fix Changelog

> **Date**: July 23, 2026  
> **Scope**: Both [finegym.ipynb](file:///c:/Users/USER/Desktop/Architecture/finegym.ipynb) and [hmdbvljepavlms.ipynb](file:///c:/Users/USER/Desktop/Architecture/hmdbvljepavlms.ipynb)  
> **Goal**: Revert the broken "enhanced" feature extraction that degraded V-JEPA 2 accuracy

---

## Problem Identified

The previous "enhancement" changed V-JEPA 2 feature extraction from **simple last-hidden-state mean pooling (1280-d)** to a **multi-layer fusion approach (10,240-d)** that concatenated the last 4 transformer layers with CLS+Mean hybrid pooling, power normalization, and L2 normalization.

> [!CAUTION]
> **This caused a massive accuracy drop** — the "enhancement" made things worse, not better.

### HMDB51 Results Comparison

| Classifier | Original (1280-d) | "Enhanced" (10,240-d) | Change |
|---|---|---|---|
| SVM-RBF Acc | **80.8%** | 66.6% | **−14.2%** |
| SVM-RBF F1 | **81.5%** | 68.4% | **−13.1%** |
| MLP Acc | **79.0%** | 67.4% | **−11.6%** |
| KNN-7 Acc | **67.2%** | 38.8% | **−28.4%** |
| Best MCC | **0.802** | 0.658 | **−0.144** |

For reference, even Qwen2.5-VL (3584-d, simple mean pooling) scored **73.1%** SVM-RBF — better than the "enhanced" V-JEPA 2 despite V-JEPA 2 being the stronger encoder.

---

## Root Cause

**Curse of dimensionality.** The 4-layer CLS+Mean fusion created 10,240-d features — 8× larger than the original 1280-d. With only ~5,400 training samples spread across 51 classes:

1. **PCA couldn't compress effectively**: PCA retained 962 dimensions (vs 286 for 1280-d input) — still too many for the training set size
2. **Distance-based classifiers degraded**: SVM-RBF and KNN rely on meaningful distances, which break down in very high dimensions
3. **Feature redundancy**: Concatenating 4 layers introduced correlated, redundant information that diluted the signal-to-noise ratio
4. **Power + L2 normalization** further compressed the dynamic range, potentially destroying discriminative magnitude information that the original features contained

---

## Fix Applied

### What Changed

Reverted V-JEPA 2 feature extraction in **both notebooks** back to the simple baseline:

```diff
- out = vjepa_model(**inputs, output_hidden_states=True)
- feat = extract_enhanced_vjepa_features(out)
+ out = vjepa_model(**inputs)
+ feat = out.last_hidden_state.mean(dim=1).cpu().float().squeeze(0)
```

### Removed
- `extract_enhanced_vjepa_features()` function
- `NUM_FUSE_LAYERS = 4` constant
- `output_hidden_states=True` argument (saves GPU memory)
- Power normalization (signed sqrt)
- L2 normalization
- Multi-layer fusion logic
- CLS+Mean hybrid pooling

### Cells Modified

| Notebook | Cell Index | Change |
|---|---|---|
| [hmdbvljepavlms.ipynb](file:///c:/Users/USER/Desktop/Architecture/hmdbvljepavlms.ipynb) | Cell 7 (markdown) | Removed "Enhanced" description |
| [hmdbvljepavlms.ipynb](file:///c:/Users/USER/Desktop/Architecture/hmdbvljepavlms.ipynb) | Cell 8 (code) | Reverted to simple mean pooling |
| [finegym.ipynb](file:///c:/Users/USER/Desktop/Architecture/finegym.ipynb) | Cell 6 (markdown) | Removed "Enhanced" description |
| [finegym.ipynb](file:///c:/Users/USER/Desktop/Architecture/finegym.ipynb) | Cell 7 (code) | Reverted to simple mean pooling |

### Expected Result After Re-run
- V-JEPA 2 features: **1280-d** (was 10,240-d)
- PCA: **1280 → ~286 dims** (was 10,240 → 962)
- SVM-RBF accuracy: **~80.8%** (was 66.6%)

> [!IMPORTANT]
> **You must delete the cached V-JEPA 2 feature files** before re-running:
> - `vjepa2_hmdb51_features.pt`
> - `vjepa2_finegym_features.pt`
> 
> Otherwise the notebook will load the old 10,240-d features from cache.

> [!NOTE]
> The Qwen2.5-VL extraction and all classifier/training code (DeepMLP, ResMLP, Conv1DNet, Transformer, AttentionPoolMLP, CosineClassifier, label smoothing, mixup) are **unchanged** — those are fine and work correctly with the 1280-d features.
