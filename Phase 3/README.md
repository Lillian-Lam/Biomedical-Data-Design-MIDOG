# Phase 3 - Uncertainty-Aware Mitosis Classification

![uncertainty distribution](https://github.com/Lillian-Lam/Biomedical-Data-Design-MIDOG/blob/main/Phase%203/results/uncertainty_distribution.png)

## Background
Extend the Phase 2 mitosis classifier with predictive-uncertainty estimation. The model uses a ResNet50 backbone with multi-scale feature extraction (layer2 + layer3 + layer4 concatenated to a 3584-d vector) and four adversarial domain classifiers (Tumor, Species, Origin, Scanner) that force the backbone to learn domain-invariant features via a Gradient Reversal Layer. We add MC Dropout to a ResNet50 backbone and run multiple stochastic forward passes per patch, so each prediction comes with a confidence score. Low-confidence predictions can be flagged for review instead of trusted blindly.

## Install
```bash
pip install torch torchvision pandas numpy pillow tqdm scikit-learn matplotlib umap-learn seaborn
```

## Usage
Inputs: the `images_split/{train,val,test}/224_patches` folders and `patch_metadata.json` files produced in Phase 2.

All files should be in the root directory before running the script:
```bash
#baseline (no domain adaptation)
python base_uncertainty.py

#full multi-domain DANN model
python uncertainty_final_model.py
```
Key knobs at the top of the file:
- `MC_DROPOUT_RATE` (default 0.3)
- `MC_NUM_FORWARD_PASSES` (default 20)
- `UNCERTAINTY_THRESHOLD` (default 0.15) - patches with predictive entropy above this are marked unreliable.

## Key files
- `base_uncertainty.py` is the baseline. It trains a plain ResNet50 with a single dropout layer inserted before the final classification head. No domain adaptation, just a clean uncertainty-aware binary classifier. Use this to establish a performance floor before running the full DANN model.
- `uncertainty_final_model.py` is the full model. It extends the baseline with the multi-domain DANN architecture: a ResNet50 backbone split into three stages (layer2 + layer3 + layer4) whose outputs are pooled separately and concatenated to a 3584-d feature vector, plus four adversarial domain classifiers (Tumor, Species, Origin, Scanner) from Phase 2.
- `document.md` - contains information on all the functions in every phase 1 script 

Results folder:
- `test_predictions_with_confidence.json` - example output: `{patch_name, true_label, predicted_label, max_probability, predictive_entropy, mutual_information, is_reliable}`.
- `result.txt` - example training log.

## Output
- Trained weights (`.pth`) and training-history plot.
- `test_predictions_with_confidence.json` - per-patch prediction and uncertainty for downstream review / triage.
