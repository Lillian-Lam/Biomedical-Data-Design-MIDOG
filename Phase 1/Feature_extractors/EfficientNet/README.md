# EfficientNet Feature Extraction

## Background
Extract patch-level features from MIDOG++ WSIs using ImageNet-pretrained EfficientNet-B0 as a generic baseline encoder for the Phase 1 domain-shift study.

## Install
```bash
pip install torch torchvision pillow tqdm pandas matplotlib umap-learn
```

## Usage
Edit `image_folder` and `output_path` at the top of `efficientnet.py`, then:
```bash
python efficientnet.py
```

## Key files
- `efficientnet.py` - tiles each WSI into 224x224 patches (tissue-filtered), runs them through EfficientNet-B0, saves features.
- `umap_efficientnet.ipynb` - 2D UMAP visualization colored by metadata.
- `umap_results_with_metadata_efficientnet.csv` - UMAP coordinates joined with slide metadata.

## Output
- `midog_efficientnet_features_patches.pkl` - patch features for downstream similarity / UMAP analysis.
