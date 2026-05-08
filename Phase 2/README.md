# Phase 2 - Mitosis Classification with Domain-Adversarial Training

## Background
Train a mitotic / non-mitotic patch classifier on MIDOG++ that generalizes across scanners, labs, species, and tumor types. We compare two ResNet50-based controls against a multi-domain DANN that adversarially suppresses scanner / lab / species / tumor cues. An optional H&E color-deconvolution variant runs the classifier on the hematoxylin channel only.

## Install
```bash
pip install torch torchvision pandas numpy pillow tqdm scikit-learn matplotlib pycocotools umap-learn
# Only needed for the deconvolution variants:
pip install histomicstk
```

## Repository Structure
```
.
├── preprocessing/
│   ├── train_test_split.py
│   ├── 224_patch_around_bbox.py
│   ├── 024_wsi_coco.json
│   └── 026_wsi_coco.json
├── control_run_cnn.py
├── control_run_cnn_deconvolution.py
├── dann_w_image_aug.py
├── dann_deconvolution.py
└── final_model.py
```

## Usage
Run scripts in this order. Edit the path constants at the top of each script.

```bash
# 1. Stratified train/test split by (Tumor, Scanner, Origin, Species) domain
python preprocessing/train_test_split.py

# 2. Crop 224x224 patches centered on each annotation bbox (writes COCO + patch_metadata.json)
python preprocessing/224_patch_around_bbox.py

# 3a. Baselines (RGB / hematoxylin)
python control_run_cnn.py
python control_run_cnn_deconvolution.py

# 3b. DANN variants (RGB with augmentation / hematoxylin via deconvolution)
python dann_w_image_aug.py
python dann_deconvolution.py

# 3c. Final model (multi-stage DANN with multi-scale features)
python final_model.py
```

## Key files

### Preprocessing (`preprocessing/`)
- `train_test_split.py` - 80/20 split per domain combination (Tumor × Scanner × Origin × Species), copies images into `images_split/`.
- `224_patch_around_bbox.py` - extracts 224×224 patches around each COCO bbox, writes `patch_metadata.json`.
- `024_wsi_coco.json`, `026_wsi_coco.json` - example COCO annotation files.

### Models
- `control_run_cnn.py` - ResNet50 baseline on RGB patches.
- `control_run_cnn_deconvolution.py` - same baseline on the hematoxylin channel after H&E color deconvolution.
- `dann_w_image_aug.py` - main multi-domain DANN: ResNet50 backbone + mitosis head + 4 adversarial domain heads (tumor, species, origin, scanner) with GRL and ramped λ schedule.
- `dann_deconvolution.py` - DANN variant on the hematoxylin channel.
- `final_model.py` - final model. A multi-domain DANN with a pretrained ResNet50 backbone split into 4 stages (stem+layer1 through layer4). Intermediate feature maps from layer2 (512-d), layer3 (1024-d), and layer4 (2048-d) are each globally average-pooled and concatenated into a 3584-d multi-scale feature vector. A mitosis classifier MLP head is trained normally on this vector, while 4 adversarial domain classifier heads (Tumor, Species, Origin, Scanner) each receive the same features through their own Gradient Reversal Layer. Lambda ramps from 0 to 2.0 over 50 epochs using the DANN paper schedule. The backbone uses a lower learning rate (1e-5) than the heads (1e-4) to preserve pretrained ImageNet features. Training augmentations include random flips, rotation, `RandomResizedCrop`, `ElasticTransform`, `ColorJitter`, shot noise, Gaussian blur, and defocus blur to simulate cross-scanner variation. Class imbalance is handled via weighted cross-entropy (mitotic weight 2.0).

## Pretrained Weights

The final model weights are available on Hugging Face: [lillianlam/multi-stage-dann-model-mitotic-figures](https://huggingface.co/lillianlam/multi-stage-dann-model-mitotic-figures)

| File | Description |
|------|-------------|
| `best_dann_model.pth` | Multi-stage DANN trained on RGB patches with augmentation |

To load the weights:

```python
import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="lillianlam/multi-stage-dann-model-mitotic-figures",
    filename="best_dann_model.pth"
)

# Reconstruct the model using final_model.py, then load weights
model = DANNModel(num_classes=2, num_domain_classes=..., lambda_val=0.0)
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()
```

## Output
Per script: trained weights (`.pth`), training-history plots, classification reports, and (for DANN) UMAP plots of the learned features colored by each domain factor.
