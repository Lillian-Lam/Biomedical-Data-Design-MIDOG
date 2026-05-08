# Phase 2 - Mitosis Classification with Domain-Adversarial Training
![Model architecture](https://github.com/Lillian-Lam/Biomedical-Data-Design-MIDOG/blob/main/Phase%202/multiscale_dann.png)
## Background
Train a mitotic / non-mitotic patch classifier on MIDOG++ that generalizes across scanners, labs, species, and tumor types. We compare two ResNet50-based controls against a multi-domain DANN that adversarially suppresses scanner / lab / species / tumor cues. An optional H&E color-deconvolution variant runs the classifier on the hematoxylin channel only.

## Install
```bash
pip install torch torchvision pandas numpy pillow tqdm scikit-learn matplotlib pycocotools umap-learn
# Only needed for cell segmentation (preprocessing):
pip install cellpose scikit-image
# Only needed for the deconvolution variants:
pip install histomicstk
```

## Repository Structure
```
.
├── preprocessing/
│   ├── cell_segmentation_to_coco.py
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
# 0. (Optional) Generate COCO annotations from unannotated WSIs
python preprocessing/cell_segmentation_to_coco.py

# 1. Stratified train/test split by (Tumor, Scanner, Origin, Species) domain
python preprocessing/train_test_split.py

# 2. Crop 224x224 patches centered on each annotation bbox
#    Run separately for train and test splits
python preprocessing/224_patch_around_bbox.py --coco_json ./images_split/train/annotations.json --image_dir ./images_split/train/ --output_dir ./images_split/train/224_patches
python preprocessing/224_patch_around_bbox.py --coco_json ./images_split/test/annotations.json --image_dir ./images_split/test/ --output_dir ./images_split/test/224_patches

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
- `cell_segmentation_to_coco.py` - optional: runs [Cellpose](https://cellpose.readthedocs.io/en/latest/index.html) cyto model on whole slide images to generate cell/nuclei bounding boxes in COCO format. Use this if you have raw WSI files without existing COCO annotations. Outputs *_wsi_coco.json files.
- `train_test_split.py` - 80/20 split per domain combination (Tumor x Scanner x Origin x Species), copies images into `images_split/`.
- `224_patch_around_bbox.py` - extracts 224×224 patches around each COCO bbox, writes `patch_metadata.json`.
- `024_wsi_coco.json`, `026_wsi_coco.json` - example COCO annotation files.

### Models
- `control_run_cnn.py` - ResNet50 baseline on RGB patches.
- `control_run_cnn_deconvolution.py` - same baseline on the hematoxylin channel after H&E color deconvolution.
- `dann_w_image_aug.py` - main multi-domain DANN: ResNet50 backbone + mitosis head + 4 adversarial domain heads (tumor, species, origin, scanner) with GRL and ramped λ schedule.
- `dann_deconvolution.py` - DANN variant on the hematoxylin channel.
- `final_model.py` - final model. A multi-domain DANN with a pretrained ResNet50 backbone split into 4 stages (stem+layer1 through layer4). Intermediate feature maps from layer2 (512-d), layer3 (1024-d), and layer4 (2048-d) are each globally average-pooled and concatenated into a 3584-d multi-scale feature vector. Training augmentations include random flips, rotation, `RandomResizedCrop`, `ElasticTransform`, `ColorJitter`, shot noise, Gaussian blur, and defocus blur to simulate cross-scanner variation.

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
    filename="best_dann_model.pth")

# Reconstruct the model using final_model.py, then load weights
model = DANNModel(num_classes=2, num_domain_classes=..., lambda_val=0.0)
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()
```

## Output
Per script: trained weights (`.pth`), training-history plots, classification reports, and (for DANN) UMAP plots of the learned features colored by each domain factor.

**For `final_model.py`**: Additional outputs include ROC curve (`auc_curve.png`), per-tumor-type F1 heatmap (`f1_heatmap_tumor.png`), and multi-panel UMAP visualization (`umap_dann.png`). All saved to `results/`.
