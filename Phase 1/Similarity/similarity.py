import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ================= CONFIGURATION =================
# 1. File Paths 
SOURCE_DIR = "../Domain Shift Quantification"
FILE_PATHS = {
    'MMD': f"{SOURCE_DIR}/mmd_unified_results/Global/MMD_domain_Global_By_Domain.csv",
    'CORAL': f"{SOURCE_DIR}/coral_unified_results/Global/CORAL_domain_Global_By_Domain.csv",
    'Wasserstein': f"{SOURCE_DIR}/wasserstein_results/Global/Wasserstein_domain_Global_By_Domain.csv"}

# 2. Weights (Must sum to 1.0)
# Strategy: Heavy emphasis on MMD (Mean), supported by Geometry (Wass) and Shape (CORAL)
WEIGHTS = {
    'MMD': 0.4,
    'Wasserstein': 0.4,
    'CORAL': 0.2
}

# 3. Gaussian Kernel Temperature (Sigma)
# Controls how "strict" the similarity is.
# Smaller sigma (e.g., 0.1) -> Only very close domains get high similarity.
# Larger sigma (e.g., 1.0) -> More generous similarity.
SIGMA = 0.5 

OUTPUT_DIR = 'similarity_matrix_output'

# ================= CORE FUNCTIONS =================

def load_matrix(csv_path):
    """
    Loads a symmetric distance matrix from a CSV file.
    Assumes the first column is the index (domain names).
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] File not found: {csv_path}")
        return None
    
    try:
        # Read CSV, setting the first column as the index
        df = pd.read_csv(csv_path, index_col=0)
        return df
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return None

def normalize_matrix(df):
    """
    Performs Min-Max Normalization on a DataFrame.
    Maps values to [0, 1] range.
    """
    matrix = df.values
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    if max_val - min_val == 0:
        return df # Avoid division by zero if all values are identical
        
    normalized = (matrix - min_val) / (max_val - min_val)
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)

def distance_to_similarity(df, sigma=0.5):
    """
    Converts a Distance Matrix to a Similarity Matrix using a Gaussian RBF Kernel.
    Similarity = exp( - distance^2 / (2 * sigma^2) )
    
    Result:
    - Distance 0 -> Similarity 1.0
    - Distance Large -> Similarity 0.0
    """
    dist_matrix = df.values
    # Apply Gaussian Kernel
    sim_matrix = np.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
    
    return pd.DataFrame(sim_matrix, index=df.index, columns=df.columns)

# ================= MAIN EXECUTION =================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- Starting Similarity Matrix Construction ---")

    # 1. Load Data
    matrices = {}
    valid_keys = []
    
    for key, path in FILE_PATHS.items():
        print(f"[INFO] Loading {key} matrix from {path}...")
        df = load_matrix(path)
        if df is not None:
            matrices[key] = df
            valid_keys.append(key)
    
    if not valid_keys:
        print("[ERROR] No matrices loaded. Exiting.")
        exit()

    # 2. Validation: Ensure all matrices have the same shape and domains
    reference_index = matrices[valid_keys[0]].index
    reference_columns = matrices[valid_keys[0]].columns
    
    for key in valid_keys:
        if not matrices[key].index.equals(reference_index):
            print(f"[ERROR] Domain mismatch in {key}! Indices do not match.")
            # Optional: You could align them here using matrices[key] = matrices[key].loc[reference_index, reference_columns]
            exit()

    # 3. Process each matrix (Normalize -> Convert to Similarity)
    similarity_matrices = {}
    
    print("\n[INFO] Processing matrices...")
    for key in valid_keys:
        # Step A: Normalize (Min-Max)
        # We normalize the DISTANCE matrix first to ensure fairness
        norm_dist_df = normalize_matrix(matrices[key])
        
        # Step B: Convert to Similarity
        sim_df = distance_to_similarity(norm_dist_df, sigma=SIGMA)
        similarity_matrices[key] = sim_df
        print(f"  -> {key}: Processed (Weight: {WEIGHTS[key]})")

    # 4. Weighted Fusion
    print("\n[INFO] Fusing matrices...")
    final_matrix = pd.DataFrame(0.0, index=reference_index, columns=reference_columns)
    
    total_weight = 0.0
    for key in valid_keys:
        w = WEIGHTS[key]
        final_matrix += similarity_matrices[key] * w
        total_weight += w
    
    # Re-normalize if weights don't sum strictly to 1 (safety check)
    if total_weight > 0:
        final_matrix /= total_weight
        
    # Ensure diagonal is exactly 1.0 (Self-Similarity)
    np.fill_diagonal(final_matrix.values, 1.0)

    # 5. Save Results
    output_csv = os.path.join(OUTPUT_DIR, 'Final_Similarity_Matrix.csv')
    final_matrix.to_csv(output_csv)
    print(f"\n[SUCCESS] Similarity Matrix saved to: {output_csv}")

    # 6. Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(final_matrix, annot=True, cmap='Greens', fmt=".2f", vmin=0, vmax=1)
    plt.title(f"Final Similarity Matrix\n(MMD={WEIGHTS['MMD']}, Wass={WEIGHTS['Wasserstein']}, CORAL={WEIGHTS['CORAL']})", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_DIR, 'Final_Similarity_Matrix.png')
    plt.savefig(output_plot, dpi=300)
    print(f"[SUCCESS] Heatmap saved to: {output_plot}")
    plt.show()
