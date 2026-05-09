# EfficientNet Feature Extraction

## Background
Extract patch-level features from MIDOG++ WSIs using ImageNet-pretrained EfficientNet-B0 as a generic baseline encoder for the Phase 1 domain-shift study.

## Install
```bash
pip install torch torchvision pillow tqdm pandas matplotlib umap-learn
```
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
Edit `image_folder` and `output_path` at the top of `efficientnet.py`, then:
```bash
python efficientnet.py
python umap_efficientnet.py
```

## Key files
- `efficientnet.py` - tiles each WSI into 224x224 patches (tissue-filtered), runs them through EfficientNet-B0, saves features.
- `umap_efficientnet.py` - 2D UMAP visualization colored by metadata.
- `umap_results_with_metadata_efficientnet.csv` - UMAP coordinates joined with slide metadata.
- `midog.csv` - slide-level metadata used to color UMAP plots.

## Output
- `midog_efficientnet_features_patches.pkl` - patch features for downstream similarity / UMAP analysis.
- umaps visualizations (our image results are also in this folder)
  
