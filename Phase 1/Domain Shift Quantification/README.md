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
- `midog.csv` - slide-level metadata
- `MMD_v1.py` - Maximum Mean Discrepancy (distribution shift).
- `CORAL_v1.py` - covariance alignment.
- `Wasserstein Distance_v1.py` - marginal transport distance.
- `Proxy A-Distance.py` - SVM-based separability score.
- `slide_505_similarity_ranking.csv` - example slide-level ranking output.

## Customization: Using Different Feature Extractors
If you wish to use a different feature extraction model (CLIP, dinoV3, etc), follow these steps:

1. Generate your feature embeddings from one of the `Phase 1/Feature_extractors`. It should save them in a .pkl dictionary format: { 'filename.tiff': {'features': np.array, ...} }.
2. In the quantification scripts (MMD_v1.py, CORAL_v1.py, etc.), update the PKL_PATH variable:
```Python
# Change this to point to your new feature file
PKL_PATH = 'path/to/your/custom_features.pkl'
```
3. Make sure your new features match the expected dimensions (e.g., 768 for CTransPath) or update the visualization scripts accordingly.

## Output
Each script writes to its own `*_results/` folder: pairwise distance CSVs and heatmap PNGs. These feed into `Phase 1/Similarity/` for fusion and ranking.
