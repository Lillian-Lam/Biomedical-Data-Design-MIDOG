import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

# ================= CONFIGURATION =================
if len(sys.argv) < 2:
    print("Usage: python MMD_v1.py <path_to_pkl_file>")
    sys.exit(1)
    
PKL_PATH = sys.argv[1]
CSV_PATH = 'midog.csv'
OUTPUT_DIR = 'coral_unified_results'

# ---------------------------------------------------------
# PART 1: CORE ALGORITHMS 
# ---------------------------------------------------------
def compute_covariance(features):
    """
    Computes the covariance matrix for a set of features.
    Input: (N_samples, N_features)
    Output: (N_features, N_features)
    """
    n = features.shape[0]
    # Need at least 2 samples to calculate variance
    if n < 2: 
        return np.zeros((features.shape[1], features.shape[1]))
    
    # np.cov expects rows as variables, so we transpose (.T)
    # ddof=1 for unbiased estimate (n-1)
    return np.cov(features.T, ddof=1)

def coral_distance(features_a, features_b):
    """
    Calculates the CORAL distance between two domains.
    Formula: ||Cov(A) - Cov(B)||^2 (Squared Frobenius Norm)
    """
    cov_a = compute_covariance(features_a)
    cov_b = compute_covariance(features_b)
    
    diff = cov_a - cov_b
    # Return the squared Frobenius norm
    return np.linalg.norm(diff, ord='fro')**2

def calculate_intra_domain_variation(features, n_splits=5):
    """
    Calculates the 'within-domain variation' (Diagonal of the matrix).
    
    Logic:
    1. Split the data into two random halves. 
    2. Calculate CORAL distance between them. 
    3. Average over 'n_splits' times. 
    """
    n = len(features)
    if n < 10:  # If too few samples, cannot effectively split
        return 0.0

    distances = []
    
    for _ in range(n_splits):
        # Randomly shuffle indices
        indices = np.random.permutation(n)
        mid = n // 2
        
        # Split into two halves
        feat_half_1 = features[indices[:mid]]
        feat_half_2 = features[indices[mid:2*mid]] # Ensure equal size
        
        # Calculate CORAL between halves
        d = coral_distance(feat_half_1, feat_half_2)
        distances.append(d)
        
    # Return the average variation
    return np.mean(distances)

# ---------------------------------------------------------
# PART 2: DATA LOADING 
# ---------------------------------------------------------
def load_and_prepare_dataframe(pkl_path, csv_path):
    """
    Loads Pickle and CSV, merging them into a single Pandas DataFrame.
    """
    if not os.path.exists(pkl_path) or not os.path.exists(csv_path):
        print(f"[ERROR] Files not found: {pkl_path} or {csv_path}")
        return None

    print(f"[INFO] Loading features from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"[INFO] Loading metadata from {csv_path}...")
    try:
        df_meta = pd.read_csv(csv_path) 
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return None
    
    # Create lookup dictionary: Slide ID -> Metadata Row
    meta_lookup = df_meta.set_index('Slide').to_dict('index')

    rows = []
    print("[INFO] Merging features with metadata...")
    
    for filename, content in data_dict.items():
        try:
            # Extract Slide ID from filename (e.g., '001.tiff' -> 1)
            base_name = Path(filename).stem
            slide_id = int(base_name)
            
            # Check if we have metadata for this slide
            if slide_id in meta_lookup:
                meta = meta_lookup[slide_id]
                feats = content['features']
                
                # Check feature validity
                if hasattr(feats, 'ndim') and feats.ndim == 2 and len(feats) > 0:
                    rows.append({
                        'filename': filename,
                        'slide_id': slide_id,
                        'domain': meta.get('domain'),   # Ensure 'domain' col exists in CSV
                        'Scanner': meta.get('Scanner'),
                        'Tumor': meta.get('Tumor'),
                        'Origin': meta.get('Origin'),
                        'features': feats # Store numpy array
                    })
        except ValueError:
            continue

    df = pd.DataFrame(rows)
    print(f"[INFO] Created DataFrame with {len(df)} slides.")
    return df

# ---------------------------------------------------------
# PART 3: UNIVERSAL ANALYSIS FUNCTION 
# ---------------------------------------------------------
def run_analysis(df, group_col, title_suffix, output_subdir=''):
    """
    Main logic to compute Matrix, Plot, and Rank.
    """
    # 1. Get unique categories
    if group_col not in df.columns:
        print(f"[WARN] Column '{group_col}' not found. Skipping.")
        return

    categories = sorted(df[group_col].dropna().unique())
    n_cats = len(categories)
    
    if n_cats < 1:
        print(f"[WARN] No categories found for {group_col}. Skipping.")
        return

    # 2. Collect features for each category
    cat_features = {}
    for cat in categories:
        subset = df[df[group_col] == cat]
        stacked = np.vstack(subset['features'].values)
        cat_features[cat] = stacked

    # 3. Compute Matrix
    matrix = np.zeros((n_cats, n_cats))
    all_pairs = [] 
    
    print(f"\n--- Computing Matrix: {title_suffix} ({n_cats} groups) ---")
    
    for i in range(n_cats):
        for j in range(n_cats):
            cat_i = categories[i]
            cat_j = categories[j]
            
            val = 0.0
            if i == j:
                # [SPLIT STRATEGY] Diagonal
                val = calculate_intra_domain_variation(cat_features[cat_i], n_splits=5)
            elif i < j:
                # [CORAL] Off-Diagonal
                val = coral_distance(cat_features[cat_i], cat_features[cat_j])
                all_pairs.append({'Pair': f"{cat_i} <-> {cat_j}", 'Distance': val})
                matrix[j, i] = val 
                matrix[i, j] = val
            
            if i > j: matrix[i, j] = matrix[j, i]
            elif i == j: matrix[i, j] = val

    # 4. Save Outputs
    save_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save CSV
    csv_name = f"CORAL_{group_col}_{title_suffix.replace(' ', '_')}.csv"
    pd.DataFrame(matrix, index=categories, columns=categories).to_csv(
        os.path.join(save_dir, csv_name)
    )
    
    # Save Heatmap
    if n_cats <= 50:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=categories, yticklabels=categories, 
                    annot=True, fmt=".2g", cmap="viridis")
        
        display_title = title_suffix 
        
        if "Global_By_" in title_suffix:
            item_name = title_suffix.split("Global_By_")[1]
            display_title = f"{item_name}s"
            
        elif "TestGroup_" in title_suffix:
            temp_name = title_suffix.replace("TestGroup_", "")
            if "_in_" in temp_name:
                item_name = temp_name.split("_in_")[0]
            else:
                item_name = temp_name
            display_title = f"{item_name}s"

        plt.title(f"CORAL Distance between {display_title} (normalized)", fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        plot_name = f"CORAL_{group_col}_{title_suffix.replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()
        print(f"[SAVE] Heatmap saved to {os.path.join(save_dir, plot_name)}")
    
    # 5. Global Ranking
    all_pairs.sort(key=lambda x: x['Distance'])
    
    print(f"\n>>> Global Top 5 Strongest Relations (Lowest Distance) [{title_suffix}]")
    print(f"{'Rank':<5} | {'Pair':<40} | {'Distance':<10}")
    print("-" * 60)
    
    top_k = min(5, len(all_pairs))
    for i in range(top_k):
        item = all_pairs[i]
        print(f"{i+1:<5} | {str(item['Pair']):<40} | {item['Distance']:.4f}")

# ---------------------------------------------------------
# PART 4: MAIN EXECUTION 
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    df = load_and_prepare_dataframe(PKL_PATH, CSV_PATH)
    
    if df is not None:
        
        # ==========================================
        # SECTION A: GLOBAL ANALYSIS 
        # Scan the entire dataset by different attributes
        # ==========================================
        print("\n" + "="*40)
        print(" SECTION A: GLOBAL ANALYSIS ")
        print("="*40)
        
        # 1. Global Domain Analysis (Strategic Selection)
        run_analysis(df, 'domain', 'Global_By_Domain', output_subdir='Global')
        
        # 2. Global Scanner Analysis (Attribution)
        run_analysis(df, 'Scanner', 'Global_By_Scanner', output_subdir='Global')
        
        # 3. Global Tumor Analysis (Attribution)
        run_analysis(df, 'Tumor', 'Global_By_Tumor', output_subdir='Global')
        
        # 4. Global Origin Analysis (Attribution)
        run_analysis(df, 'Origin', 'Global_By_Origin', output_subdir='Global')
        
        # 5. Global WSI Analysis 
        # ==========================================
        # SECTION B: SPECIFIC TEST GROUPS 
        # Controlled experiments defined in MMD file
        # ==========================================
        print("\n" + "="*40)
        print(" SECTION B: SPECIFIC TEST GROUPS ")
        print("="*40)
        
        # Test 1: Tumor Type (Within 'Hamamatsu XR' scanner only)
        # Why: Eliminate scanner bias to see pure tumor differences
        subset_1 = df[df['Scanner'] == 'Hamamatsu XR']
        if not subset_1.empty:
            run_analysis(subset_1, 'Tumor', 'TestGroup_Tumor_in_Hamamatsu', output_subdir='TestGroups')

        # Test 2: Scanner (Within 'human breast cancer' only)
        # Why: Eliminate tumor bias to see pure scanner differences
        subset_2 = df[df['Tumor'] == 'human breast cancer']
        if not subset_2.empty:
            run_analysis(subset_2, 'Scanner', 'TestGroup_Scanner_in_BreastCancer', output_subdir='TestGroups')

        # Test 3: Origin (Within 'canine soft tissue sarcoma' only)
        # Why: Eliminate other factors to see Lab Origin differences
        subset_3 = df[df['Tumor'] == 'canine soft tissue sarcoma']
        if not subset_3.empty:
            run_analysis(subset_3, 'Origin', 'TestGroup_Origin_in_CanineSarcoma', output_subdir='TestGroups')
            
        print("\n[DONE] All analyses complete. Check 'coral_unified_results' folder.")
