import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc  
from pathlib import Path

# ================= CONFIGURATION =================
PKL_PATH = '../Feature_extractors/results_norm/midog_features_patches_normalized.pkl' 
CSV_PATH = 'midog.csv'
OUTPUT_DIR = 'mmd_unified_results'
# ========================================================

# ---------------------------------------------------------
# PART 1: CORE ALGORITHMS (Memory Optimized MMD)
# ---------------------------------------------------------

def mmd_unbiased(X, Y, use_scaled=True):
    """
    Computes the unbiased estimator of MMD using a Linear Kernel.
    Unlike the naive implementation that computes a (N x N) Gram matrix (causing OOM),
    this implementation uses the algebraic expansion of the Frobenius norm:
    Sum(X dot X.T) = ||Sum(X)||^2 - Sum(||X||^2)
    
    Time Complexity: O(N * d) instead of O(N^2 * d)
    Space Complexity: O(d) instead of O(N^2)
    """
    n, m = len(X), len(Y)

    if use_scaled:
        # Scale with pooled statistics
        # Compute stats without concatenating huge arrays
        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        var_X = np.var(X)
        var_Y = np.var(Y)
        
        pooled_mean = (mean_X * n + mean_Y * m) / (n + m)
        # Combined variance formula
        pooled_var = ((var_X + mean_X**2) * n + (var_Y + mean_Y**2) * m) / (n + m) - pooled_mean**2
        pooled_std = np.sqrt(pooled_var)
        
        scale_factor = pooled_std if pooled_std > 0 else 1.0
        X = X / scale_factor
        Y = Y / scale_factor

    # --- Optimized Linear Kernel Calculation (No Huge Matrices) ---
    
    # 1. Term XX: Sum of off-diagonal elements of K(X, X)
    # Formula: ||sum(X)||^2 - sum(||x_i||^2)
    sum_X = np.sum(X, axis=0)
    sum_sq_norm_X = np.sum(np.sum(X ** 2, axis=1))
    sum_XX_off_diag = np.dot(sum_X, sum_X) - sum_sq_norm_X

    # 2. Term YY: Sum of off-diagonal elements of K(Y, Y)
    sum_Y = np.sum(Y, axis=0)
    sum_sq_norm_Y = np.sum(np.sum(Y ** 2, axis=1))
    sum_YY_off_diag = np.dot(sum_Y, sum_Y) - sum_sq_norm_Y

    # 3. Term XY: Sum of all elements of K(X, Y)
    # Formula: dot(sum(X), sum(Y))
    sum_XY = np.dot(sum_X, sum_Y)

    # Unbiased Estimator Formula
    mmd_sq = (sum_XX_off_diag) / (n * (n - 1)) + \
             (sum_YY_off_diag) / (m * (m - 1)) - \
             2 * (sum_XY) / (n * m)

    return mmd_sq

def mmd_intra_domain(X, n_splits=5, use_scaled=True):
    """
    Calculates 'Intra-domain MMD' (Diagonal) by splitting data into halves.
    """
    if len(X) < 10:
        return 0.0  

    mmds = []
    for _ in range(n_splits):
        # Random permutation
        indices = np.random.permutation(len(X))
        mid = len(X) // 2
        
        # Split into two halves
        X1 = X[indices[:mid]]
        X2 = X[indices[mid:2*mid]]  # Ensure equal sizes

        # Use the memory-optimized function
        mmd = mmd_unbiased(X1, X2, use_scaled=use_scaled)
        mmds.append(mmd)

    return np.mean(mmds)

# ---------------------------------------------------------
# PART 2: DATA LOADING 
# ---------------------------------------------------------
def load_and_prepare_dataframe(pkl_path, csv_path):
    """
    Standardized data loader with error handling for CSV separators.
    """
    if not os.path.exists(pkl_path) or not os.path.exists(csv_path):
        print(f"[ERROR] Files not found: {pkl_path} or {csv_path}")
        return None

    print(f"[INFO] Loading features from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"[INFO] Loading metadata from {csv_path}...")
    try:
        # 1. Try default (comma)
        df_meta = pd.read_csv(csv_path) 
        # 2. If 'Slide' column is missing, try semicolon (fallback)
        if 'Slide' not in df_meta.columns:
             print("[WARN] 'Slide' column missing, trying semicolon separator...")
             df_meta = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return None
    
    # --- Data Cleaning ---
    df_meta.columns = df_meta.columns.str.strip()
    df_meta['Slide'] = df_meta['Slide'].astype(str).str.strip()
    df_meta = df_meta.replace('Hamammatsu XR', 'Hamamatsu XR') # Fix Typo
    
    if 'Dataset' in df_meta.columns:
        df_meta = df_meta[df_meta['Dataset'] == 'train'].copy()

    meta_lookup = df_meta.set_index('Slide').to_dict('index')
    rows = []
    print("[INFO] Merging features with metadata...")
    
    for filename, content in data_dict.items():
        try:
            base_name = Path(filename).stem
            slide_id = str(int(base_name)) 
            
            if slide_id in meta_lookup:
                meta = meta_lookup[slide_id]
                feats = content['features'] if isinstance(content, dict) and 'features' in content else content
                
                if hasattr(feats, 'ndim') and feats.ndim == 2 and len(feats) > 0:
                    rows.append({
                        'filename': filename,
                        'slide_id': slide_id,
                        'domain': meta.get('domain'), # Make sure this col exists in CSV
                        'Scanner': meta.get('Scanner'),
                        'Tumor': meta.get('Tumor'),
                        'Origin': meta.get('Origin'),
                        'features': feats 
                    })
        except ValueError:
            continue

    # Clean up memory immediately
    del data_dict
    gc.collect()

    df = pd.DataFrame(rows)
    print(f"[INFO] Created DataFrame with {len(df)} slides.")
    return df

# ---------------------------------------------------------
# PART 3: UNIVERSAL ANALYSIS FUNCTION
# ---------------------------------------------------------
def run_mmd_analysis(df, group_col, title_suffix, output_subdir=''):
    """
    Main logic to compute MMD Matrix, Plot, and Rank.
    """
    # 1. Check Column
    if group_col not in df.columns:
        print(f"[WARN] Column '{group_col}' not found. Skipping.")
        return

    categories = sorted(df[group_col].dropna().unique())
    n_cats = len(categories)
    
    if n_cats < 1:
        print(f"[WARN] No categories found for {group_col}. Skipping.")
        return

    # 2. Pre-collect features (Memory & Speed Optimization)
    cat_features = {}
    print(f"[INFO] Aggregating features for {group_col}...")
    for cat in categories:
        subset = df[df[group_col] == cat]
        if not subset.empty:
            cat_features[cat] = np.vstack(subset['features'].values)
        else:
            cat_features[cat] = np.empty((0, 768))

    # 3. Compute Matrix
    matrix = np.zeros((n_cats, n_cats))
    all_pairs = [] 
    
    print(f"\n--- Computing MMD Matrix: {title_suffix} ({n_cats} groups) ---")
    
    for i in range(n_cats):
        for j in range(n_cats):
            cat_i = categories[i]
            cat_j = categories[j]
            
            val = 0.0
            
            # Optimization: Calculate mainly upper triangle
            if i <= j:
                feat_i = cat_features[cat_i]
                feat_j = cat_features[cat_j]

                if len(feat_i) == 0 or len(feat_j) == 0:
                    val = 0.0
                elif i == j:
                    # Diagonal
                    val = mmd_intra_domain(feat_i, n_splits=5, use_scaled=True)
                else:
                    # Off-Diagonal
                    val = mmd_unbiased(feat_i, feat_j, use_scaled=True)
                
                # Store for ranking
                if i != j:
                    all_pairs.append({'Pair': f"{cat_i} <-> {cat_j}", 'MMD': val})
                
                # Symmetric filling
                matrix[i, j] = val
                matrix[j, i] = val
                
                # GC periodically
                if j % 5 == 0: gc.collect()

    # 4. Save Outputs
    save_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save CSV
    csv_name = f"MMD_{group_col}_{title_suffix.replace(' ', '_')}.csv"
    pd.DataFrame(matrix, index=categories, columns=categories).to_csv(
        os.path.join(save_dir, csv_name)
    )
    
    # Save Heatmap
    if n_cats <= 50:
        plt.figure(figsize=(10, 8))
        # MMD values are small, use .4f
        sns.heatmap(matrix, xticklabels=categories, yticklabels=categories, 
                    annot=True, fmt=".4f", cmap="YlOrRd")
        
        display_title = title_suffix
        if "Global_By_" in title_suffix:
            if len(title_suffix.split("Global_By_")) > 1:
                display_title = f"{title_suffix.split('Global_By_')[1]}s"
        elif "TestGroup_" in title_suffix:
            display_title = title_suffix.replace("TestGroup_", "")

        plt.title(f"MMD Distance: {display_title}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        plot_name = f"MMD_{group_col}_{title_suffix.replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()
        print(f"[SAVE] Heatmap saved to {os.path.join(save_dir, plot_name)}")
    
    # 5. Global Ranking
    all_pairs.sort(key=lambda x: x['MMD'])
    
    print(f"\n>>> Top 5 Closest Relations (Lowest MMD) [{title_suffix}]")
    print(f"{'Rank':<5} | {'Pair':<40} | {'MMD':<10}")
    print("-" * 60)
    
    top_k = min(5, len(all_pairs))
    for i in range(top_k):
        item = all_pairs[i]
        print(f"{i+1:<5} | {str(item['Pair']):<40} | {item['MMD']:.4f}")

# ---------------------------------------------------------
# PART 4: MAIN EXECUTION 
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    df = load_and_prepare_dataframe(PKL_PATH, CSV_PATH)
    
    if df is not None:
        
        # ==========================================
        # SECTION A: GLOBAL ANALYSIS
        # ==========================================
        print("\n" + "="*40)
        print(" SECTION A: GLOBAL ANALYSIS (MMD) ")
        print("="*40)
        
        # [NEW]: Added Domain Analysis as requested
        run_mmd_analysis(df, 'domain', 'Global_By_Domain', output_subdir='Global')
        
        # Standard Analyses
        run_mmd_analysis(df, 'Scanner', 'Global_By_Scanner', output_subdir='Global')
        run_mmd_analysis(df, 'Tumor', 'Global_By_Tumor', output_subdir='Global')
        run_mmd_analysis(df, 'Origin', 'Global_By_Origin', output_subdir='Global')
        
        # ==========================================
        # SECTION B: SPECIFIC TEST GROUPS
        # ==========================================
        print("\n" + "="*40)
        print(" SECTION B: SPECIFIC TEST GROUPS (MMD) ")
        print("="*40)
        
        # Test 1: Tumor Type (Hamamatsu XR only)
        subset_1 = df[df['Scanner'] == 'Hamamatsu XR']
        if not subset_1.empty:
            run_mmd_analysis(subset_1, 'Tumor', 'TestGroup_Tumor_in_Hamamatsu', output_subdir='TestGroups')

        # Test 2: Scanner (Breast Cancer only)
        subset_2 = df[df['Tumor'] == 'human breast cancer']
        if not subset_2.empty:
            run_mmd_analysis(subset_2, 'Scanner', 'TestGroup_Scanner_in_BreastCancer', output_subdir='TestGroups')

        # Test 3: Origin (Canine Sarcoma only)
        subset_3 = df[df['Tumor'] == 'canine soft tissue sarcoma']
        if not subset_3.empty:
            run_mmd_analysis(subset_3, 'Origin', 'TestGroup_Origin_in_CanineSarcoma', output_subdir='TestGroups')
            
        print("\n[DONE] All MMD analyses complete. Check 'mmd_unified_results' folder.")
