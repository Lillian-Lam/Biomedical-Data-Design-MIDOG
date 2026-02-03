import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import wasserstein_distance  # Key import for original formula

# ================= CONFIGURATION =================
PKL_PATH = 'midog_features_patches_normalized(cyclegan).pkl'
CSV_PATH = 'midog.csv'
OUTPUT_DIR = 'wasserstein_results'

# ---------------------------------------------------------
# PART 1: CORE ALGORITHMS
# ---------------------------------------------------------

def compute_wasserstein_distance(features_a, features_b):
    """
    Calculates the Average Marginal Wasserstein Distance (W1) between two domains.
    
    Original Formula Logic:
    For 1D distributions u and v, the first Wasserstein distance is defined as:
    l_1_distance(CDF_u, CDF_v).
    
    For high-dimensional data (Multivariate), we compute the W1 distance 
    for each dimension independently and take the average (Marginal Wasserstein).
    
    Input: (N_samples, N_features) matrices
    Output: Scalar distance
    """
    # 1. Check dimensions
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError("Feature dimensions do not match!")
        
    n_features = features_a.shape[1]
    total_dist = 0.0
    
    # 2. Compute W1 for each feature dimension 
    for i in range(n_features):
        u_values = features_a[:, i]
        v_values = features_b[:, i]
        total_dist += wasserstein_distance(u_values, v_values)
        
    # 3. Return average distance across all dimensions
    return total_dist / n_features

def calculate_intra_domain_variation(features, n_splits=5):
    """
    Calculates the 'within-domain variation' using Wasserstein Distance.

    Logic:
    1. Split the data into two random halves.
    2. Calculate Wasserstein distance between them.
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
        
        # Calculate Wasserstein between halves
        d = compute_wasserstein_distance(feat_half_1, feat_half_2)
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
                        'domain': meta.get('domain'),
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
    Main logic to compute Matrix, Plot, and Rank using Wasserstein Distance.
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
        # Stack all features in this category
        stacked = np.vstack(subset['features'].values)
        cat_features[cat] = stacked

    # 3. Compute Matrix
    matrix = np.zeros((n_cats, n_cats))
    all_pairs = [] 
    
    print(f"\n--- Computing Wasserstein Matrix: {title_suffix} ({n_cats} groups) ---")
    print("Note: Computing Wasserstein for high dimensions may take a moment...")
    
    for i in range(n_cats):
        for j in range(n_cats):
            cat_i = categories[i]
            cat_j = categories[j]
            
            val = 0.0
            if i == j:
                # [SPLIT STRATEGY] Diagonal: Internal Variation
                val = calculate_intra_domain_variation(cat_features[cat_i], n_splits=5)
            elif i < j:
                # [Wasserstein] Off-Diagonal: Distance
                val = compute_wasserstein_distance(cat_features[cat_i], cat_features[cat_j])
                
                # Store for ranking
                all_pairs.append({'Pair': f"{cat_i} <-> {cat_j}", 'Distance': val})
                
                # Symmetric filling
                matrix[j, i] = val 
                matrix[i, j] = val
            
            if i > j: matrix[i, j] = matrix[j, i]
            elif i == j: matrix[i, j] = val

    # 4. Save Outputs
    save_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save CSV
    csv_name = f"Wasserstein_{group_col}_{title_suffix.replace(' ', '_')}.csv"
    pd.DataFrame(matrix, index=categories, columns=categories).to_csv(
        os.path.join(save_dir, csv_name)
    )
    
    # Save Heatmap
    if n_cats <= 50:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=categories, yticklabels=categories, 
                    annot=True, fmt=".4f", cmap="viridis")
        
        # Title Generation 
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

        plt.title(f"Wasserstein Distance between {display_title} (normalized)", fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        plot_name = f"Wasserstein_{group_col}_{title_suffix.replace(' ', '_')}.png"
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
        # ==========================================
        print("\n" + "="*40)
        print(" SECTION A: GLOBAL ANALYSIS (Wasserstein) ")
        print("="*40)
        
        run_analysis(df, 'domain', 'Global_By_Domain', output_subdir='Global')
        run_analysis(df, 'Scanner', 'Global_By_Scanner', output_subdir='Global')
        run_analysis(df, 'Tumor', 'Global_By_Tumor', output_subdir='Global')
        run_analysis(df, 'Origin', 'Global_By_Origin', output_subdir='Global')

        # ==========================================
        # SECTION B: SPECIFIC TEST GROUPS
        # ==========================================
        print("\n" + "="*40)
        print(" SECTION B: SPECIFIC TEST GROUPS (Wasserstein) ")
        print("="*40)
        
        # Test 1: Tumor Type (Hamamatsu XR only)
        subset_1 = df[df['Scanner'] == 'Hamamatsu XR']
        if not subset_1.empty:
            run_analysis(subset_1, 'Tumor', 'TestGroup_Tumor_in_Hamamatsu', output_subdir='TestGroups')

        # Test 2: Scanner (Breast Cancer only)
        subset_2 = df[df['Tumor'] == 'human breast cancer']
        if not subset_2.empty:
            run_analysis(subset_2, 'Scanner', 'TestGroup_Scanner_in_BreastCancer', output_subdir='TestGroups')

        # Test 3: Origin (Canine Sarcoma only)
        subset_3 = df[df['Tumor'] == 'canine soft tissue sarcoma']
        if not subset_3.empty:
            run_analysis(subset_3, 'Origin', 'TestGroup_Origin_in_CanineSarcoma', output_subdir='TestGroups')
            
        print("\n[DONE] All Wasserstein analyses complete.")