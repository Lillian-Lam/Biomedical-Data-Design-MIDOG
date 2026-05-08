# CLIP Feature Extraction

## Background
Extract patch-level features from MIDOG++ WSIs using OpenAI CLIP (ViT-B/32) as a baseline encoder for the Phase 1 domain-shift study.

## Install
```bash
pip install torch torchvision pillow tqdm pandas matplotlib
pip install git+https://github.com/openai/CLIP.git
```

## Usage
Edit `image_folder` and `output_path` at the top of `clipnet.py`, then:
```bash
python clipnet.py
python clip.py
```

## Key file
- `clipnet.py` - tiles each WSI into 224x224 patches (tissue-filtered), runs them through CLIP ViT-B/32, and saves features to a pickle.
- `umap_clip.py` - makes the 2D UMAP visualization of the embeddings.
- `midog.csv` - slide-level metadata used to color UMAP plots.

## Output
- `midog_clip_features_patches.pkl` - dict of `{slide_id: features}`, fed downstream into UMAP / similarity matrices.
- umaps visualizations (our image results are also in this folder)
