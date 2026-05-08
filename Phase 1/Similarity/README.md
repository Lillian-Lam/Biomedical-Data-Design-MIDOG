# Similarity Matrix

## Background
Fuse the MMD / CORAL / Wasserstein distance matrices from `Phase 1/Domain Shift Quantification` into a single similarity matrix. Two granularities are supported: domain-level (groups of slides) and WSI-level (individual slides). The fused matrix drives the Phase 2 strategic-selection / training-weighting decisions.

## Install
```bash
pip install numpy pandas scipy seaborn matplotlib
```

## Usage
Run from inside this folder.

Domain-level fusion (reads pre-computed metric CSVs):
```bash
python similarity.py
```
Domain-level Weighted Raw Shift (Distance):
```bash
python domain_shift_score.py
```
WSI-level fusion (reads a feature pickle, computes all three metrics from scratch, then fuses):
```bash
python similarity_wsi.py
```
Edit the `WEIGHTS`, `SIGMA`, and file paths at the top of each script.

## Method
Similarity Fusion (`similarity.py`)
1. Load (or compute) per-metric pairwise distance matrices.
2. Min-max normalize each matrix to [0, 1].
3. Convert distance to similarity with a Gaussian kernel `exp(-d^2 / 2σ^2)`.
4. Weighted fuse: `MMD * 0.5 + Wasserstein * 0.3 + CORAL * 0.2` (default in `similarity_wsi.py`).

Raw Shift Scoring (`domain_shift_score.py`)
1. Accesses raw CSVs from mmd_unified_results, coral_unified_results, and wasserstein_results.
2. Weighted fuse: `MMD * 0.5 + Wasserstein * 0.3 + CORAL * 0.2` of raw distances

## Key files
- `midog.csv` - slide-level metadata
- `domain_shift_score.py` - Fuses raw distances into a weighted shift score.
- `similarity.py` - domain-level fusion into [0, 1] similarity
- `similarity_wsi.py` - WSI-level fusion direct from a feature pickle.
  
Results folder:
- `Final_Similarity_Matrix.csv` / `WSI_Fused_Similarity_Matrix.csv` - example outputs.
- `slide_505_similarity_ranking.csv` - example similarity ranking for slide 505.

## Output
`similarity.py` / `similarity_wsi.py`: A fused similarity matrix CSV plus a heatmap PNG in `similarity_matrix_output/` or `wsi_fusion_results/`.
`domain_shift_score.py`: heatmaps with specific control titles (e.g., "Scanner Control", "Tumor Type Control").
