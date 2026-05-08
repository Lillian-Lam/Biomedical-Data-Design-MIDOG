# Biomedical-Data-Design-MIDOG++

## Background
Mitotic count is a key tumor-grading signal but it is labor-intensive and variable across observers. Automating it is hard because feature appearance shifts with scanner, lab, species, and tumor type, so a model can easily learn acquisition cues instead of biology. This repo addresses that in three phases.

## Data Acquisition
This project uses the MIDOG++ dataset. To run the pipeline:

Download the images and metadata from the official [MIDOG++](https://github.com/DeepMicroscopy/MIDOGpp) GitHub repository. 
```bash
git clone https://github.com/Xiyue-Wang/TransPath.git](https://github.com/DeepMicroscopy/MIDOGpp
```

## Pipeline

### [Phase 1 - Domain Shift Quantification](Phase%201/)
Extract patch features with five different encoders (CTransPath, CLIP, EfficientNet, DINOv3, MAE) and quantify domain shift across scanners / labs / tumor types using MMD, CORAL, and Wasserstein distances. Fuse the three metrics into one similarity matrix.
- `Feature_extractors/` - one sub-folder per encoder.
- `Domain Shift Quantification/` - per-metric distance matrices.
- `Similarity/` - weighted fusion (domain-level and WSI-level).

### [Phase 2 - Domain-Adversarial Classifier](Phase%202/)
Train a ResNet50 mitotic / non-mitotic patch classifier with multi-domain DANN (4 adversarial heads: tumor, species, origin, scanner) and compare against RGB and hematoxylin-only baselines.

### [Phase 3 - Uncertainty-Aware Inference](Phase%203/)
Add MC Dropout to the Phase 2 classifier and report per-patch confidence so low-confidence predictions can be flagged.

## Install
```bash
pip install torch torchvision pandas numpy pillow tqdm scikit-learn matplotlib seaborn scipy umap-learn pycocotools timm
# Phase 2 deconvolution variants
pip install histomicstk
# Phase 2 all cell annotations in WSI
pip install cellpose
# Phase 1 / CTransPath
git clone https://github.com/Xiyue-Wang/TransPath.git
git clone https://github.com/DBO-DKFZ/multistain_cyclegan_normalization.git
```
Python 3.9+, GPU recommended. Each phase has its own README with detailed steps.

## Usage
Run the phases in order. Each script has path / hyperparameter constants at the top - edit those before running.
```bash
# Phase 1: pick an encoder, extract features, then quantify and fuse
python "Phase 1/Feature_extractors/CTransPath/ctranspath_cycleGAN_norm.py"
python "Phase 1/Domain Shift Quantification/MMD_v1.py"
python "Phase 1/Similarity/similarity_wsi.py"

# Phase 2: split, patch, train
python "Phase 2/preprocessing/train_test_split.py"
python "Phase 2/preprocessing/224_patch_around_bbox.py"
python "Phase 2/final_model.py"

# Phase 3: train with MC Dropout, write per-patch confidence
python "Phase 3/base_uncertainty.py"
```

## Repo layout
- `Phase 1/`, `Phase 2/`, `Phase 3/` - see each folder's README.
- `Presentation/` - dated slide decks from class check-ins with results of code output.
- `requirements.txt` - Python dependency list.
- `LICENSE`
