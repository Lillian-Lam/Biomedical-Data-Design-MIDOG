# Domain Shift Quantification

## Background
Quantify domain shift in MIDOG++ patch features across scanners, tumor types, and lab origins. Each metric measures the gap between two feature distributions in a different way, letting us compare how strongly each metadata axis biases the features.

## Install
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```
Inputs: `midog.csv` and a feature pickle from Phase 1 (e.g. `midog_feature_patches.pkl`). Place both alongside the script.

## Usage
Run from inside this folder so relative paths resolve. 
```bash
python MMD_v1.py                    
python CORAL_v1.py
python "Wasserstein Distance_v1.py"
python "Proxy A-Distance.py"
```
Each script first does a global pass (by `domain` / `Scanner` / `Tumor` / `Origin`), then a controlled pass that fixes one factor and varies another.

## Key files
- `MMD_v1.py` - Maximum Mean Discrepancy (distribution shift).
- `CORAL_v1.py` - covariance alignment.
- `Wasserstein Distance_v1.py` - marginal transport distance.
- `Proxy A-Distance.py` - SVM-based separability score.
- `slide_505_similarity_ranking.csv` - example slide-level ranking output.

## Output
Each script writes to its own `*_results/` folder: pairwise distance CSVs and heatmap PNGs. These feed into `Phase 1/Similarity/` for fusion and ranking.
