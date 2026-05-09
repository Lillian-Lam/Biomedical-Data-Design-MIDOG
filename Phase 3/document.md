\# Phase 3 - Function Documentation



\## Overview

This document provides function-level documentation for Phase 3 scripts, which add Monte Carlo (MC) Dropout uncertainty quantification to the Phase 2 DANN classifier. Many functions are reused from Phase 2 (`final\_model.py`). Only new or modified functions are documented here.



\*\*Reused from Phase 2\*\* (see Phase 2 documentation for details):

\- `GradientReversalFunction`, `GradientReversalLayer`

\- `ShotNoise`, `DefocusBlur`

\- `MitosisDataset` (with domain labels)

\- `collate\_fn`

\- `DANNModel` (multi-scale, 3584-d features)

\- `get\_lambda`, `train\_one\_epoch`, `evaluate`

\- `plot\_training\_history`, `plot\_auc`, `plot\_f1\_heatmap`, `plot\_umap`, `extract\_features`



\## 1. base\_uncertainty.py - Baseline CNN with MC Dropout



\### `SimpleCNN\_MCDropout.\_\_init\_\_(num\_classes=2, dropout\_rate=MC\_DROPOUT\_RATE)`

\*\*Purpose\*\*: Initialize ResNet50 with dropout before the final classification layer for MC Dropout.



\*\*Args\*\*:

\- `num\_classes` (int): Number of output classes (default: 2)

\- `dropout\_rate` (float): Dropout probability for MC sampling (default: 0.3)



\*\*Returns\*\*: None



\*\*Architecture\*\*:

```python

self.backbone.fc = nn.Sequential(

&#x20;   nn.Dropout(p=dropout\_rate),

&#x20;   nn.Linear(in\_features, num\_classes)

)

```



\### `SimpleCNN\_MCDropout.forward(x)`

\*\*Purpose\*\*: Standard forward pass (dropout active only during training by default).



\*\*Args\*\*:

\- `x` (torch.Tensor): Input tensor, shape (B, 3, 224, 224)



\*\*Returns\*\*: `torch.Tensor` - Logits, shape (B, num\_classes)





\### `enable\_mc\_dropout(model)`

\*\*Purpose\*\*: Set model to eval mode but re-enable dropout layers for stochastic inference.



\*\*Args\*\*:

\- `model` (nn.Module): PyTorch model with Dropout layers



\*\*Returns\*\*: None



\*\*Behavior\*\*:

\- Calls `model.eval()` to freeze BatchNorm

\- Iterates through all modules and sets `Dropout` layers to `train()` mode



\### `mc\_dropout\_predict(model, images, n\_forward=MC\_NUM\_FORWARD\_PASSES)`

\*\*Purpose\*\*: Run multiple stochastic forward passes and aggregate predictions.



\*\*Args\*\*:

\- `model` (nn.Module): Model with dropout layers

\- `images` (torch.Tensor): Input batch, shape (B, 3, 224, 224)

\- `n\_forward` (int): Number of stochastic forward passes (default: 20)



\*\*Returns\*\*:

\- `tuple`: `(mean\_probs, predictive\_entropy, mutual\_info)` where:

&#x20; - `mean\_probs` (torch.Tensor): Averaged softmax across passes, shape (B, 2)

&#x20; - `predictive\_entropy` (torch.Tensor): Total uncertainty H(E\[p]), shape (B,)

&#x20; - `mutual\_info` (torch.Tensor): Epistemic (model) uncertainty, shape (B,)



\*\*Formulas\*\*:

\- `mean\_probs = average(softmax(logits\_i))`

\- `predictive\_entropy = -sum(mean\_probs \* log(mean\_probs))`

\- `expected\_entropy = average(-sum(probs\_i \* log(probs\_i)))`

\- `mutual\_info = predictive\_entropy - expected\_entropy`





\### `evaluate\_with\_uncertainty(model, loader, device, n\_forward=MC\_NUM\_FORWARD\_PASSES, uncertainty\_threshold=UNCERTAINTY\_THRESHOLD)`

\*\*Purpose\*\*: Run MC Dropout inference over entire dataset and split into reliable/ambiguous subsets.



\*\*Args\*\*:

\- `model` (nn.Module): Trained model with dropout

\- `loader` (DataLoader): Test data loader

\- `device` (torch.device): 'cuda' or 'cpu'

\- `n\_forward` (int): Number of stochastic passes (default: 20)

\- `uncertainty\_threshold` (float): Entropy cutoff for reliable vs ambiguous (default: 0.15)



\*\*Returns\*\*:

\- `tuple`: `(all\_preds, all\_labels, all\_entropy, all\_mutual\_info, all\_max\_prob)` where:

&#x20; - `all\_preds` (np.ndarray): Predicted labels (0/1)

&#x20; - `all\_labels` (np.ndarray): Ground truth labels

&#x20; - `all\_entropy` (np.ndarray): Predictive entropy per sample

&#x20; - `all\_mutual\_info` (np.ndarray): Mutual information per sample

&#x20; - `all\_max\_prob` (np.ndarray): Maximum softmax probability per sample



\*\*Console Output\*\*:

\- Total samples, reliable/ambiguous counts and percentages

\- Mean entropy, mutual info, max probability

\- Classification reports for: All samples, Reliable subset, Ambiguous subset



\*\*Interpretation\*\*:

\- `entropy < threshold`: Reliable prediction (trust model)

\- `entropy >= threshold`: Ambiguous (flag for pathologist review)





\### `plot\_uncertainty\_distribution(all\_entropy, all\_labels, all\_preds, threshold=UNCERTAINTY\_THRESHOLD, save\_path='uncertainty\_distribution.png')`

\*\*Purpose\*\*: Create two-panel histogram of uncertainty distribution.



\*\*Args\*\*:

\- `all\_entropy` (np.ndarray): Predictive entropy values

\- `all\_labels` (np.ndarray): Ground truth labels

\- `all\_preds` (np.ndarray): Predicted labels

\- `threshold` (float): Uncertainty threshold (default: 0.15)

\- `save\_path` (str): Output path for figure



\*\*Returns\*\*: None



\*\*Panels\*\*:

| Panel | Content | Purpose |

|-------|---------|---------|

| Left | Histogram by true label (Non-mitotic vs Mitotic) | See if mitotic patches are inherently more uncertain |

| Right | Histogram by correctness (Correct vs Incorrect) | Confirm that high entropy correlates with errors |



\*\*Output\*\*: Saves `uncertainty\_distribution.png`



\## 2. uncertainty\_final\_model.py - Multi-Stage DANN with MC Dropout



This script combines the multi-scale DANN from Phase 2 (`final\_model.py`) with MC Dropout uncertainty quantification. Key differences from Phase 2:



\### Modified `DANNModel.\_\_init\_\_()`

\*\*Change\*\*: Dropout added to mitosis classifier head:



```python

self.mitosis\_classifier = nn.Sequential(

&#x20;   nn.Flatten(),

&#x20;   nn.Linear(feature\_dim, 512),

&#x20;   nn.ReLU(),

&#x20;   nn.Dropout(p=MC\_DROPOUT\_RATE),  # ← MC Dropout layer

&#x20;   nn.Linear(512, num\_classes)

)

```



\*\*MC\_DROPOUT\_RATE\*\*: 0.3 (default)





\### `enable\_mc\_dropout(model)` (same as base\_uncertainty.py)

\*\*Purpose\*\*: Enable dropout during inference while keeping BatchNorm frozen.



\### `mc\_dropout\_predict(model, images, n\_forward=MC\_NUM\_FORWARD\_PASSES)`

\*\*Purpose\*\*: Run stochastic passes through \*\*mitosis head only\*\* (domain heads skipped).



\*\*Key difference from Phase 2\*\*: Uses `model.predict\_only()` to bypass domain classifiers.



\*\*Args\*\*:

\- `model` (DANNModel): Multi-scale DANN model

\- `images` (torch.Tensor): Input batch

\- `n\_forward` (int): Number of stochastic passes (default: 20)



\*\*Returns\*\*: Same as base\_uncertainty version



\*\*Process\*\*:

1\. Enable MC Dropout

2\. For each pass: `logits = model.predict\_only(images)` (no domain heads)

3\. Softmax, aggregate, compute entropy and mutual info



\### `evaluate\_with\_uncertainty(model, loader, device, n\_forward=MC\_NUM\_FORWARD\_PASSES, uncertainty\_threshold=UNCERTAINTY\_THRESHOLD)`

\*\*Purpose\*\*: Extended version that also collects domain labels.



\*\*Returns\*\*:

\- `tuple`: `(all\_preds, all\_labels, all\_entropy, all\_mutual\_info, all\_max\_prob, all\_domain\_label\_list)`



\*\*Addition\*\*: `all\_domain\_label\_list` (dict) - domain labels per sample for cross-referencing uncertainty by tumor type.



\### `plot\_uncertainty\_distribution()` (same as base\_uncertainty.py)



\### Modified `main()`

\*\*Purpose\*\*: Full training pipeline + MC Dropout uncertainty evaluation.



\*\*New steps after training\*\* (in addition to Phase 2 outputs):

1\. Deterministic evaluation (baseline)

2\. MC Dropout uncertainty evaluation

3\. Uncertainty distribution plot

4\. Per-sample JSON with confidence scores



\*\*Output files\*\* (saved to `results/`):

| File | Description |

|------|-------------|

| `best\_dann\_model.pth` | Best model weights (from Phase 2 training) |

| `training\_history.png` | 7-panel training curves |

| `auc\_curve.png` | ROC curve with AUC |

| `f1\_heatmap\_tumor.png` | Per-tumor-type F1 scores |

| `umap\_dann.png` | 5-panel UMAP visualization |

| `uncertainty\_distribution.png` | NEW: Two-panel entropy histogram |

| `test\_predictions\_with\_confidence.json` | NEW: Per-sample predictions with confidence scores |



\## 3. Common Data Structures



\### JSON Output Format (`test\_predictions\_with\_confidence.json`)



```json

\[

&#x20;   {

&#x20;       "patch\_name": "001\_ann0.tif",

&#x20;       "true\_label": 1,

&#x20;       "predicted\_label": 1,

&#x20;       "max\_probability": 0.92,

&#x20;       "predictive\_entropy": 0.08,

&#x20;       "mutual\_information": 0.03,

&#x20;       "is\_reliable": true

&#x20;   },

&#x20;   ...

]

```



| Field | Description |

|-------|-------------|

| `patch\_name` | Filename of the patch |

| `true\_label` | Ground truth (1=mitotic, 0=non-mitotic) |

| `predicted\_label` | Model prediction |

| `max\_probability` | Highest softmax probability |

| `predictive\_entropy` | Total uncertainty (H) |

| `mutual\_information` | Epistemic/model uncertainty |

| `is\_reliable` | `True` if entropy < threshold |


Gemini was used to help format and draft the documentation.md based on my original code. I reviewed and edited all descriptions for technical accuracy.


