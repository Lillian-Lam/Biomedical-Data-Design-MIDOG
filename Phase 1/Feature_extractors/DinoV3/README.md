# DINOv3 Feature Extraction

## Background
Extract patch-level features from MIDOG++ WSIs using a self-supervised DINOv3 ViT-Base encoder for the Phase 1 domain-shift study.

## Install
```bash
pip install torch torchvision timm pillow tqdm
```
The script sets `HF_ENDPOINT=https://hf-mirror.com` for those who cannot reach huggingface.co directly. Remove that line if not needed.

## Usage
extract features and plot umap
```bash
python dinov3.py /path/to/your/midog_images
python umap_dinov3.py
```

## Key files
- `dinov3.py` - tiles WSIs into 224x224 patches (tissue-filtered), runs them through `vit_base_patch16_dinov3.lvd1689m`, saves features.
- `umap_dinov3.py` - makes the 2D UMAP visualization of the embeddings.
- `midog.csv` - slide-level metadata used to color UMAP plots.

## Output
- `midog_dinov3_features_patches.pkl` - patch features for downstream similarity / UMAP analysis.
- umaps visualizations (our image results are also in this folder)
