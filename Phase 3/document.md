# Phase 3 - Function Documentation

## Overview

This document provides function-level documentation for Phase 3 scripts, which add Monte Carlo (MC) Dropout uncertainty quantification to the Phase 2 DANN classifier. Many functions are reused from Phase 2 (`final_model.py`). Only new or modified functions are documented here.

**Reused from Phase 2** (see Phase 2 documentation for details):

- `GradientReversalFunction`, `GradientReversalLayer`
- `ShotNoise`, `DefocusBlur`
- `MitosisDataset` (with domain labels)
- `collate_fn`
- `DANNModel` (multi-scale, 3584-d features)
- `get_lambda`, `train_one_epoch`, `evaluate`
- `plot_training_history`, `plot_auc`, `plot_f1_heatmap`, `plot_umap`, `extract_features`

## 1. base_uncertainty.py - Baseline CNN with MC Dropout

### `SimpleCNN_MCDropout.__init__(num_classes=2, dropout_rate=MC_DROPOUT_RATE)`

**Purpose**: Initialize ResNet50 with dropout before the final classification layer for MC Dropout.

**Args**:
- `num_classes` (int): Number of output classes (default: 2)
- `dropout_rate` (float): Dropout probability for MC sampling (default: 0.3)

**Returns**: None

**Architecture**:
```python
self.backbone.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(in_features, num_classes)
)
```

### `SimpleCNN_MCDropout.forward(x)`

**Purpose**: Standard forward pass (dropout active only during training by default).

**Args**:
- `x` (torch.Tensor): Input tensor, shape (B, 3, 224, 224)

**Returns**: `torch.Tensor` - Logits, shape (B, num_classes)

### `enable_mc_dropout(model)`

**Purpose**: Set model to eval mode but re-enable dropout layers for stochastic inference.

**Args**:
- `model` (nn.Module): PyTorch model with Dropout layers

**Returns**: None

**Behavior**:
- Calls `model.eval()` to freeze BatchNorm
- Iterates through all modules and sets `Dropout` layers to `train()` mode

### `mc_dropout_predict(model, images, n_forward=MC_NUM_FORWARD_PASSES)`

**Purpose**: Run multiple stochastic forward passes and aggregate predictions.

**Args**:
- `model` (nn.Module): Model with dropout layers
- `images` (torch.Tensor): Input batch, shape (B, 3, 224, 224)
- `n_forward` (int): Number of stochastic forward passes (default: 20)

**Returns**:
- `tuple`: `(mean_probs, predictive_entropy, mutual_info)` where:
  - `mean_probs` (torch.Tensor): Averaged softmax across passes, shape (B, 2)
  - `predictive_entropy` (torch.Tensor): Total uncertainty H(E[p]), shape (B,)
  - `mutual_info` (torch.Tensor): Epistemic (model) uncertainty, shape (B,)

**Formulas**:
- `mean_probs = average(softmax(logits_i))`
- `predictive_entropy = -sum(mean_probs * log(mean_probs))`
- `expected_entropy = average(-sum(probs_i * log(probs_i)))`
- `mutual_info = predictive_entropy - expected_entropy`

### `evaluate_with_uncertainty(model, loader, device, n_forward=MC_NUM_FORWARD_PASSES, uncertainty_threshold=UNCERTAINTY_THRESHOLD)`

**Purpose**: Run MC Dropout inference over entire dataset and split into reliable/ambiguous subsets.

**Args**:
- `model` (nn.Module): Trained model with dropout
- `loader` (DataLoader): Test data loader
- `device` (torch.device): 'cuda' or 'cpu'
- `n_forward` (int): Number of stochastic passes (default: 20)
- `uncertainty_threshold` (float): Entropy cutoff for reliable vs ambiguous (default: 0.15)

**Returns**:
- `tuple`: `(all_preds, all_labels, all_entropy, all_mutual_info, all_max_prob)` where:
  - `all_preds` (np.ndarray): Predicted labels (0/1)
  - `all_labels` (np.ndarray): Ground truth labels
  - `all_entropy` (np.ndarray): Predictive entropy per sample
  - `all_mutual_info` (np.ndarray): Mutual information per sample
  - `all_max_prob` (np.ndarray): Maximum softmax probability per sample

**Console Output**:
- Total samples, reliable/ambiguous counts and percentages
- Mean entropy, mutual info, max probability
- Classification reports for: All samples, Reliable subset, Ambiguous subset

**Interpretation**:
- `entropy < threshold`: Reliable prediction (trust model)
- `entropy >= threshold`: Ambiguous (flag for pathologist review)

### `plot_uncertainty_distribution(all_entropy, all_labels, all_preds, threshold=UNCERTAINTY_THRESHOLD, save_path='uncertainty_distribution.png')`

**Purpose**: Create two-panel histogram of uncertainty distribution.

**Args**:
- `all_entropy` (np.ndarray): Predictive entropy values
- `all_labels` (np.ndarray): Ground truth labels
- `all_preds` (np.ndarray): Predicted labels
- `threshold` (float): Uncertainty threshold (default: 0.15)
- `save_path` (str): Output path for figure

**Returns**: None

**Panels**:

| Panel | Content | Purpose |
|-------|---------|---------|
| Left | Histogram by true label (Non-mitotic vs Mitotic) | See if mitotic patches are inherently more uncertain |
| Right | Histogram by correctness (Correct vs Incorrect) | Confirm that high entropy correlates with errors |

**Output**: Saves `uncertainty_distribution.png`

## 2. uncertainty_final_model.py - Multi-Stage DANN with MC Dropout

This script combines the multi-scale DANN from Phase 2 (`final_model.py`) with MC Dropout uncertainty quantification. Key differences from Phase 2:

### Modified `DANNModel.__init__()`

**Change**: Dropout added to mitosis classifier head:

```python
self.mitosis_classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(feature_dim, 512),
    nn.ReLU(),
    nn.Dropout(p=MC_DROPOUT_RATE),  # ← MC Dropout layer
    nn.Linear(512, num_classes)
)
```

**MC_DROPOUT_RATE**: 0.3 (default)

### `enable_mc_dropout(model)` (same as base_uncertainty.py)

**Purpose**: Enable dropout during inference while keeping BatchNorm frozen.

### `mc_dropout_predict(model, images, n_forward=MC_NUM_FORWARD_PASSES)`

**Purpose**: Run stochastic passes through **mitosis head only** (domain heads skipped).

**Key difference from Phase 2**: Uses `model.predict_only()` to bypass domain classifiers.

**Args**:
- `model` (DANNModel): Multi-scale DANN model
- `images` (torch.Tensor): Input batch
- `n_forward` (int): Number of stochastic passes (default: 20)

**Returns**: Same as base_uncertainty version

**Process**:
1. Enable MC Dropout
2. For each pass: `logits = model.predict_only(images)` (no domain heads)
3. Softmax, aggregate, compute entropy and mutual info

### `evaluate_with_uncertainty(model, loader, device, n_forward=MC_NUM_FORWARD_PASSES, uncertainty_threshold=UNCERTAINTY_THRESHOLD)`

**Purpose**: Extended version that also collects domain labels.

**Returns**:
- `tuple`: `(all_preds, all_labels, all_entropy, all_mutual_info, all_max_prob, all_domain_label_list)`

**Addition**: `all_domain_label_list` (dict) - domain labels per sample for cross-referencing uncertainty by tumor type.

### `plot_uncertainty_distribution()` (same as base_uncertainty.py)

### Modified `main()`

**Purpose**: Full training pipeline + MC Dropout uncertainty evaluation.

**New steps after training** (in addition to Phase 2 outputs):
1. Deterministic evaluation (baseline)
2. MC Dropout uncertainty evaluation
3. Uncertainty distribution plot
4. Per-sample JSON with confidence scores

**Output files** (saved to `results/`):

| File | Description |
|------|-------------|
| `best_dann_model.pth` | Best model weights (from Phase 2 training) |
| `training_history.png` | 7-panel training curves |
| `auc_curve.png` | ROC curve with AUC |
| `f1_heatmap_tumor.png` | Per-tumor-type F1 scores |
| `umap_dann.png` | 5-panel UMAP visualization |
| `uncertainty_distribution.png` | NEW: Two-panel entropy histogram |
| `test_predictions_with_confidence.json` | NEW: Per-sample predictions with confidence scores |

## 3. Common Data Structures

### JSON Output Format (`test_predictions_with_confidence.json`)

```json
[
    {
        "patch_name": "001_ann0.tif",
        "true_label": 1,
        "predicted_label": 1,
        "max_probability": 0.92,
        "predictive_entropy": 0.08,
        "mutual_information": 0.03,
        "is_reliable": true
    }
]
```

| Field | Description |
|-------|-------------|
| `patch_name` | Filename of the patch |
| `true_label` | Ground truth (1=mitotic, 0=non-mitotic) |
| `predicted_label` | Model prediction |
| `max_probability` | Highest softmax probability |
| `predictive_entropy` | Total uncertainty (H) |
| `mutual_information` | Epistemic/model uncertainty |
| `is_reliable` | `True` if entropy < threshold |


Gemini was used to help format and draft the documentation.md based on my original code. I reviewed and edited all descriptions for technical accuracy.


