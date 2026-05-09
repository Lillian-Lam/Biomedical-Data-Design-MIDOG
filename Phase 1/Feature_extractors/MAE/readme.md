# MAE Feature Extraction

## Background
Extract patch-level features from MIDOG++ WSIs using a self-supervised MAE ViT-Base encoder for the Phase 1 domain-shift study.

## Install
```bash
pip install torch torchvision timm pillow tqdm pandas
```
The script sets `HF_ENDPOINT=https://hf-mirror.com` for those who cannot reach huggingface.co directly. 

## Configuration: Setting Your Image Path
Before running any extraction script, you must update the path to where your MIDOG++ images are stored on your local machine.

1. Open the script you wish to run.
2. Locate the Configuration section at the top.
3. Update the image_folder variable:

```Python
# CHANGE THIS: Point to your local folder containing .tiff files
image_folder = '/path/to/your/local/MIDOGpp/images'
```

## Usage
Edit `image_folder` and `output_path` at the top of `mae.py`, then:
```bash
python mae.py
python umap_mae.py
```

## Key files
- `mae.py` - tiles WSIs into 224x224 patches (tissue-filtered), runs them through `mae_vit_base_patch16`, saves features.
- `umap_mae.py` - 2D UMAP visualization of the embeddings.
- `midog.csv` - slide-level metadata used to color UMAP plots.
- `umap_results_with_metadata_mae.csv` - UMAP coordinates joined with metadata.

## Output
- `midog_mae_features_patches.pkl` - patch features for downstream similarity / UMAP analysis.
- umaps visualizations (our image results are also in this folder)
