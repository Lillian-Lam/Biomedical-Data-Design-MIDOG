import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc
from pathlib import Path
from scipy.stats import wasserstein_distance

# ================= CONFIGURATION =================
#change this if you are using a different feature extractor
PKL_PATH = '../Feature_extractors/results_norm/midog_features_patches_normalized.pkl'
CSV_PATH = 'midog.csv'
OUTPUT_DIR = 'wsi_fusion_results'

# Weight configuration based on team discussion
# Logic: MMD (Mean) > Wasserstein (Geometry) > CORAL (Covariance)
WEIGHTS = {
    'MMD': 0.5,
    'Wasserstein': 0.3,
    'CORAL': 0.2
}

# Gaussian Kernel Sigma (Temperature parameter)
# Controls the sensitivity of converting distance to similarity.
# Lower sigma makes the similarity drop faster as distance increases.
SIGMA = 0.5 

# Sampling limit to speed up WSI-level calculation
# Calculating Wasserstein on >5000 points is computationally expensive.
# 2000 patches are usually sufficient to represent the slide distribution.
MAX_PATCHES_PER_SLIDE = 2000 
# =================================================

# ---------------------------------------------------------
# PART 1: METRIC FUNCTIONS
# ---------------------------------------------------------

def mmd_linear(X, Y):
    """ 
    Computes MMD using a Linear Kernel with O(N) complexity optimization.
    
    Mathematical Logic:
    Instead of calculating the full N*N Gram matrix, we use the identity:
    || Sum(X) ||^2 = Sum(X dot X.T)
    This avoids Out-Of-Memory (OOM) errors on large WSI feature sets.
    """
    n, m = len(X), len(Y)
    if n == 0 or m == 0: return 1.0
    
    # Calculate terms based on sum of vectors and sum of squared norms
    sum_X = np.sum(X, axis=0)
    sum_sq_norm_X = np.sum(np.sum(X ** 2, axis=1))
    sum_XX = np.dot(sum_X, sum_X) - sum_sq_norm_X

    sum_Y = np.sum(Y, axis=0)
    sum_sq_norm_Y = np.sum(np.sum(Y ** 2, axis=1))
    sum_YY = np.dot(sum_Y, sum_Y) - sum_sq_norm_Y

    sum_XY = np.dot(sum_X, sum_Y)

    # Unbiased estimate of MMD^2
    mmd_sq = (sum_XX)/(n*(n-1)) + (sum_YY)/(m*(m-1)) - 2*(sum_XY)/(n*m)
    return mmd_sq

def coral_dist(X, Y):
    """ 
    Computes CORAL distance (Correlation Alignment).
    Measure: Squared Frobenius norm of the difference between covariance matrices.
    Focus: Captures 'style' or 'texture' second-order statistics.
    """
    if len(X) < 2 or len(Y) < 2: return 0.0
    
    # Compute covariance matrices (rowvar=False implies columns are features)
    cov_a = np.cov(X.T)
    cov_b = np.cov(Y.T)
    
    diff = cov_a - cov_b
    return np.linalg.norm(diff, ord='fro')**2

def wasserstein_marginal(X, Y):
    """ 
    Computes Marginal Wasserstein Distance (Earth Mover's Distance).
    Logic: Averages the 1D Wasserstein distance across all 768 feature dimensions.
    Focus: Geometric discrepancy between distributions.
    """
    n_feats = X.shape[1]
    total_dist = 0.0
    # Iterate over each feature dimension
    for i in range(n_feats):
        total_dist += wasserstein_distance(X[:, i], Y[:, i])
    return total_dist / n_feats

# ---------------------------------------------------------
# PART 2: HELPER FUNCTIONS
# ---------------------------------------------------------

def normalize_matrix(mat):
    """ Performs Global Min-Max Normalization to scale values to [0, 1]. """
    min_val, max_val = np.min(mat), np.max(mat)
    if max_val - min_val == 0: return np.zeros_like(mat)
    return (mat - min_val) / (max_val - min_val)

def dist_to_sim(mat, sigma=0.5):
    """ 
    Converts a Distance Matrix to a Similarity Matrix using a Gaussian Kernel.
    Formula: Similarity = exp( - distance^2 / (2 * sigma^2) )
    Result: 0 distance -> 1.0 similarity.
    """
    return np.exp(- (mat ** 2) / (2 * sigma ** 2))

def load_data(pkl, csv):
    """ Loads Pickle features and CSV metadata, grouping by Slide ID. """
    if not os.path.exists(pkl) or not os.path.exists(csv):
        print("Files not found."); return None
    
    print(f"[INFO] Loading features from {pkl}...")
    with open(pkl, 'rb') as f: data = pickle.load(f)
    
    print(f"[INFO] Loading metadata from {csv}...")
    try:
        # Handle potential separator differences (comma vs semicolon)
        meta = pd.read_csv(csv)
        if 'Slide' not in meta.columns: meta = pd.read_csv(csv, sep=';')
    except Exception as e:
        print(f"[ERROR] CSV Read Failed: {e}"); return None
    
    # Clean Slide IDs to standard string format
    meta.columns = meta.columns.str.strip()
    meta['Slide'] = meta['Slide'].astype(str).str.strip()
    
    # Filter for Train dataset only
    if 'Dataset' in meta.columns: meta = meta[meta['Dataset'] == 'train']
    
    meta_dict = meta.set_index('Slide').to_dict('index')
    
    slide_features = {}
    print("[INFO] Grouping features by Slide ID...")
    
    for fname, content in data.items():
        try:
            # Extract ID from filename (e.g., '001.tiff' -> '1')
            base = str(int(Path(fname).stem))
            
            if base in meta_dict:
                feat = content['features'] if isinstance(content, dict) else content
                if feat.ndim == 2 and len(feat) > 0:
                    if base not in slide_features: slide_features[base] = []
                    slide_features[base].append(feat)
        except: continue
        
    del data; gc.collect() # Free memory
    
    # Stack patches into single arrays for each slide
    final_features = {}
    for sid, feats in slide_features.items():
        stacked = np.vstack(feats)
        
        # Optimization: Downsample if too many patches
        if len(stacked) > MAX_PATCHES_PER_SLIDE:
            idx = np.random.choice(len(stacked), MAX_PATCHES_PER_SLIDE, replace=False)
            stacked = stacked[idx]
            
        final_features[sid] = stacked
        
    print(f"[INFO] Successfully loaded {len(final_features)} slides.")
    return final_features

# ---------------------------------------------------------
# PART 3: MAIN FUSION LOGIC
# ---------------------------------------------------------

def generate_fused_matrix(slide_feats):
    """ 
    Main function to compute, normalize, and fuse similarity matrices.
    """
    # 1. Sort Slide IDs numerically (1, 2, ... 10) instead of alphabetically (1, 10, 2)
    sorted_ids = sorted(slide_feats.keys(), key=lambda x: int(x))
    n = len(sorted_ids)
    
    # 2. Initialize Empty Matrices for each metric
    mat_mmd = np.zeros((n, n))
    mat_coral = np.zeros((n, n))
    mat_wass = np.zeros((n, n))
    
    print(f"\n--- Computing 3 Metrics for {n} Slides (O(N^2)) ---")
    print("Note: This process may take time depending on CPU power.")
    
    # 3. Compute All Metrics (Pairwise iteration)
    for i in range(n):
        if i % 5 == 0: 
            print(f"  Processing row {i}/{n} (Slide {sorted_ids[i]})...")
            gc.collect() # Periodic garbage collection
        
        for j in range(n):
            if i <= j: # Calculate Upper Triangle
                feat_i = slide_feats[sorted_ids[i]]
                feat_j = slide_feats[sorted_ids[j]]
                
                if i == j:
                    # Diagonal (Self-distance) is always 0
                    d_mmd, d_coral, d_wass = 0.0, 0.0, 0.0
                else:
                    # Calculate raw distances
                    d_mmd = mmd_linear(feat_i, feat_j)
                    d_coral = coral_dist(feat_i, feat_j)
                    d_wass = wasserstein_marginal(feat_i, feat_j)
                
                # Fill matrices symmetrically
                mat_mmd[i, j] = mat_mmd[j, i] = d_mmd
                mat_coral[i, j] = mat_coral[j, i] = d_coral
                mat_wass[i, j] = mat_wass[j, i] = d_wass
    
    # 4. Normalize (Global Min-Max) -> Map to [0, 1] range
    print("\n[INFO] Normalizing raw distance matrices...")
    norm_mmd = normalize_matrix(mat_mmd)
    norm_coral = normalize_matrix(mat_coral)
    norm_wass = normalize_matrix(mat_wass)
    
    # 5. Convert Distance to Similarity (Gaussian Kernel) -> Map to [1, 0]
    print("[INFO] Converting distances to similarities (Gaussian Kernel)...")
    sim_mmd = dist_to_sim(norm_mmd, sigma=SIGMA)
    sim_coral = dist_to_sim(norm_coral, sigma=SIGMA)
    sim_wass = dist_to_sim(norm_wass, sigma=SIGMA)
    
    # 6. Weighted Fusion
    print(f"[INFO] Fusing matrices with weights: {WEIGHTS}")
    final_sim = (WEIGHTS['MMD'] * sim_mmd + 
                 WEIGHTS['Wasserstein'] * sim_wass + 
                 WEIGHTS['CORAL'] * sim_coral)
    
    # Ensure diagonal is exactly 1.0 (Perfect self-similarity)
    np.fill_diagonal(final_sim, 1.0)
    
    return final_sim, sorted_ids

# ---------------------------------------------------------
# PART 4: EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load Data
    feats = load_data(PKL_PATH, CSV_PATH)
    
    if feats:
        # Run Computation
        matrix, ids = generate_fused_matrix(feats)
        
        # Save CSV
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_out = pd.DataFrame(matrix, index=ids, columns=ids)
        
        csv_path = os.path.join(OUTPUT_DIR, "WSI_Fused_Similarity_Matrix.csv")
        df_out.to_csv(csv_path)
        print(f"\n[SUCCESS] Matrix saved to: {csv_path}")
        
        # Plot Heatmap
        plt.figure(figsize=(12, 10))
        # Turn off tick labels if there are too many slides to avoid clutter
        sns.heatmap(df_out, cmap="Greens", square=True, xticklabels=False, yticklabels=False)
        plt.title(f"Fused WSI Similarity Matrix\n(MMD={WEIGHTS['MMD']}, Wass={WEIGHTS['Wasserstein']}, CORAL={WEIGHTS['CORAL']})")
        plt.xlabel(f"Slide Index (Total {len(ids)})")
        plt.ylabel("Slide Index")
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, "WSI_Fused_Heatmap.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[SUCCESS] Heatmap saved to: {plot_path}")
        print("[DONE] Processing complete.")
