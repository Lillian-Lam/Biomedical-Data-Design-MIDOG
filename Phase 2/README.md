# Phase 2 - Mitosis Classification with Domain-Adversarial Training

## Background
Train a mitotic / non-mitotic patch classifier on MIDOG++ that generalizes across scanners, labs, species, and tumor types. We compare two ResNet50-based controls against a multi-domain DANN that adversarially suppresses scanner / lab / species / tumor cues. An optional H&E color-deconvolution variant runs the classifier on the hematoxylin channel only.

## Install
```bash
pip install torch torchvision pandas numpy pillow tqdm scikit-learn matplotlib pycocotools umap-learn
# Only needed for the deconvolution variants:
pip install histomicstk
```

## Usage
Run scripts in this order. Edit the path constants at the top of each script.

```bash
# 1. Stratified train/test split by (Tumor, Scanner, Origin, Species) domain
python train_test_split.py

# 2. Crop 224x224 patches centered on each annotation bbox (writes COCO + patch_metadata.json)
python 224_patch_around_bbox.py

# 3a. Baselines (RGB / hematoxylin)
python control_run_cnn.py
python control_run_cnn_deconvolution.py

# 3b. DANN (RGB with augmentation / hematoxylin via deconvolution)
python dann_image_aug.py
python DANN/dann_deconvolution.py
```

## Key files
- `train_test_split.py` - 80/20 split per domain combination (Tumor × Scanner × Origin × Species), copies images into `images_split/`.
- `224_patch_around_bbox.py` - extracts 224×224 patches around each COCO bbox, writes `patch_metadata.json`.
- `024_wsi_coco.json`, `026_wsi_coco.json` - example COCO annotation files.
- `control_run_cnn.py` - ResNet50 baseline on RGB patches.
- `control_run_cnn_deconvolution.py` - same baseline on the hematoxylin channel after H&E color deconvolution.
- `dann_image_aug.py` - main multi-domain DANN: ResNet50 backbone + mitosis head + 4 adversarial domain heads (tumor, species, origin, scanner) with GRL and ramped λ schedule.
- `DANN/dann_deconvolution.py` - DANN variant on the hematoxylin channel.

## Output
Per script: trained weights (`.pth`), training-history plots, classification reports, and (for DANN) UMAP plots of the learned features colored by each domain factor.
