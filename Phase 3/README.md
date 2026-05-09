# Phase 3 - Uncertainty-Aware Mitosis Classification

## Background
Extend the Phase 2 mitosis classifier with predictive-uncertainty estimation. We add MC Dropout to a ResNet50 backbone and run multiple stochastic forward passes per patch, so each prediction comes with a confidence score. Low-confidence predictions can be flagged for review instead of trusted blindly.

## Install
```bash
pip install torch torchvision pandas numpy pillow tqdm scikit-learn matplotlib
```
Inputs: the `images_split/{train,val,test}/224_patches` folders and `patch_metadata.json` files produced in Phase 2.

## Usage
Edit the path constants at the top of the script, then:
```bash
python base_uncertainty.py
```
Key knobs at the top of the file:
- `MC_DROPOUT_RATE` (default 0.3)
- `MC_NUM_FORWARD_PASSES` (default 20)
- `UNCERTAINTY_THRESHOLD` (default 0.15) - patches with predictive entropy above this are marked unreliable.

## Key files
- `base_uncertainty.py` - training + MC-Dropout inference. Trains the classifier, then runs N stochastic passes on the test set and writes per-patch confidence scores.

Results folder:
- `test_predictions_with_confidence.json` - example output: `{patch_name, true_label, predicted_label, max_probability, predictive_entropy, mutual_information, is_reliable}`.
- `result.txt` - example training log.

## Output
- Trained weights (`.pth`) and training-history plot.
- `test_predictions_with_confidence.json` - per-patch prediction and uncertainty for downstream review / triage.
