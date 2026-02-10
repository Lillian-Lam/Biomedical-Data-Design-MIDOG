import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Machine Learning Imports for Proxy A-Distance
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress SVM convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ================= CONFIGURATION =================
PKL_PATH = 'midog_feature_patches.pkl'
CSV_PATH = 'midog.csv'
OUTPUT_DIR = 'proxy_a_distance_results'

# ---------------------------------------------------------
# PART 1: CORE ALGORITHMS (Proxy A-Distance)
# ---------------------------------------------------------

def compute_proxy_a_distance(features_source, features_target, cv_folds=5):
    """
    Computes the Proxy A-Distance (PAD) between two domains.
    
    Theory (Ben-David et al., 2010):
    PAD = 2 * (1 - 2 * epsilon)
    where 'epsilon' is the generalization error of a classifier (e.g., Linear SVM)
    trained to discriminate between the two domains.
    
    Interpretation:
    - PAD = 0: Domains are indistinguishable (Classifier Acc ~ 50%). Good alignment.
    - PAD = 2: Domains are distinct (Classifier Acc ~ 100%). Large Shift.
    
    Input:
        features_source: (N_samples_a, N_features)
        features_target: (N_samples_b, N_features)
        cv_folds: Number of cross-validation folds
    Output:
        Scalar distance (0.0 to 2.0)
    """
    # 1. Create Labels (0 for Source, 1 for Target)
    n_source = len(features_source)
    n_target = len(features_target)
    
    # Safety check for small datasets
    if n_source < cv_folds or n_target < cv_folds:
        print(f"  [WARN] Not enough samples for {cv_folds}-fold CV ({n_source}, {n_target}). Returning NaN.")
        return np.nan

    X = np.vstack((features_source, features_target))
    y = np.hstack((np.zeros(n_source), np.ones(n_target)))
    
    # 2. Define Classifier (Linear SVM is standard for PAD)
    # We use a Pipeline to include Scaling, as SVM is sensitive to scale
    clf = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42, C=1.0))
    
    # 3. Compute Accuracy using Cross-Validation
    # We use Stratified K-Fold to ensure class balance in folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    try:
        # Get accuracy scores (1 - error)
        accuracies = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        mean_acc = np.mean(accuracies)
        
        # 4. Calculate PAD
        # Formula: d_A = 2 * (1 - 2 * error)
        # error = 1 - accuracy
        # => d_A = 2 * (1 - 2 * (1 - acc)) = 2 * (2 * acc - 1)
        
        # If accuracy < 0.5 (worse than random), we clip PAD to 0
        pad_value = 2.0 * (2.0 * mean_acc - 1.0)
        pad_value = max(0.0, pad_value) 
        
        return pad_value
        
    except Exception as e:
        print(f"  [ERROR] CV Failed: {e}")
        return 0.0

def calculate_intra_domain_pad(features, cv_folds=5):
    """
    Calculates Self-PAD (Diagonal) by splitting the domain into two halves.
    Expected value is close to 0.0 (indistinguishable).
    """
    n = len(features)
    if n < 10: return 0.0
    
    # Randomly shuffle and split into two pseudo-domains
    indices = np.random.permutation(n)
    mid = n // 2
    
    feat_half_1 = features[indices[:mid]]
    feat_half_2 = features[indices[mid:2*mid]] # Ensure roughly equal size
    
    return compute_proxy_a_distance(feat_half_1, feat_half_2, cv_folds=cv_folds)

# ---------------------------------------------------------
# PART 2: DATA LOADING 
# ---------------------------------------------------------
def load_and_prepare_dataframe(pkl_path, csv_path):
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
    
    meta_lookup = df_meta.set_index('Slide').to_dict('index')
    rows = []
    print("[INFO] Merging features with metadata...")
    
    for filename, content in data_dict.items():
        try:
            base_name = Path(filename).stem
            slide_id = int(base_name)
            
            if slide_id in meta_lookup:
                meta = meta_lookup[slide_id]
                feats = content['features']
                
                if hasattr(feats, 'ndim') and feats.ndim == 2 and len(feats) > 0:
                    rows.append({
                        'filename': filename,
                        'slide_id': slide_id,
                        'domain': meta.get('domain'),
                        'Scanner': meta.get('Scanner'),
                        'Tumor': meta.get('Tumor'),
                        'Origin': meta.get('Origin'),
                        'features': feats 
                    })
        except ValueError:
            continue

    df = pd.DataFrame(rows)
    print(f"[INFO] Created DataFrame with {len(df)} slides.")
    return df

# ---------------------------------------------------------
# PART 3: UNIVERSAL ANALYSIS FUNCTION 
# ---------------------------------------------------------
def run_pad_analysis(df, group_col, title_suffix, output_subdir=''):
    """
    Main logic to compute PAD Matrix, Plot, and Rank.
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
    print(f"[INFO] Aggregating features for {group_col}...")
    for cat in categories:
        subset = df[df[group_col] == cat]
        stacked = np.vstack(subset['features'].values)
        cat_features[cat] = stacked

    # 3. Compute Matrix
    matrix = np.zeros((n_cats, n_cats))
    all_pairs = [] 
    
    print(f"\n--- Computing Proxy A-Distance Matrix: {title_suffix} ({n_cats} groups) ---")
    print("Note: Training SVMs for PAD calculation. This may take a moment...")
    
    for i in range(n_cats):
        for j in range(n_cats):
            cat_i = categories[i]
            cat_j = categories[j]
            
            val = 0.0
            
            # Optimization: PAD matrix is symmetric, calculate mainly upper triangle
            if i <= j:
                if i == j:
                    # Diagonal: Split single domain into two
                    val = calculate_intra_domain_pad(cat_features[cat_i])
                else:
                    # Off-Diagonal: Domain A vs Domain B
                    val = compute_proxy_a_distance(cat_features[cat_i], cat_features[cat_j])
                
                # Store
                if i != j:
                    all_pairs.append({'Pair': f"{cat_i} <-> {cat_j}", 'PAD': val})
                
                # Symmetric filling
                matrix[i, j] = val
                matrix[j, i] = val
            
            # (Progress indicator mostly for large matrices)
            if n_cats > 5 and j == n_cats - 1:
                print(f"  Processed row {i+1}/{n_cats}")

    # 4. Save Outputs
    save_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save CSV
    csv_name = f"PAD_{group_col}_{title_suffix.replace(' ', '_')}.csv"
    pd.DataFrame(matrix, index=categories, columns=categories).to_csv(
        os.path.join(save_dir, csv_name)
    )
    
    # Save Heatmap
    if n_cats <= 50:
        plt.figure(figsize=(10, 8))
        # PAD is 0 to 2. Let's fix vmin/vmax for consistent colorbar
        sns.heatmap(matrix, xticklabels=categories, yticklabels=categories, 
                    annot=True, fmt=".2f", cmap="magma_r", vmin=0, vmax=2.0)
        
        # Dynamic Title Logic
        display_title = title_suffix
        if "Global_By_" in title_suffix:
            display_title = f"{title_suffix.split('Global_By_')[1]}s"
        elif "TestGroup_" in title_suffix:
            display_title = title_suffix.replace("TestGroup_", "")

        plt.title(f"Proxy A-Distance: {display_title}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        plot_name = f"PAD_{group_col}_{title_suffix.replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()
        print(f"[SAVE] Heatmap saved to {os.path.join(save_dir, plot_name)}")
    
    # 5. Global Ranking (Higher PAD = More Distinct = Worse Adaptation)
    all_pairs.sort(key=lambda x: x['PAD'], reverse=True)
    
    print(f"\n>>> Top 5 Largest Domain Shifts (Highest PAD) [{title_suffix}]")
    print(f"{'Rank':<5} | {'Pair':<40} | {'PAD':<10}")
    print("-" * 60)
    
    top_k = min(5, len(all_pairs))
    for i in range(top_k):
        item = all_pairs[i]
        print(f"{i+1:<5} | {str(item['Pair']):<40} | {item['PAD']:.4f}")

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
        print(" SECTION A: GLOBAL ANALYSIS (Proxy A-Distance) ")
        print("="*40)
        
        run_pad_analysis(df, 'domain', 'Global_By_Domain', output_subdir='Global')
        run_pad_analysis(df, 'Scanner', 'Global_By_Scanner', output_subdir='Global')
        run_pad_analysis(df, 'Tumor', 'Global_By_Tumor', output_subdir='Global')
        run_pad_analysis(df, 'Origin', 'Global_By_Origin', output_subdir='Global')

        # ==========================================
        # SECTION B: SPECIFIC TEST GROUPS
        # ==========================================
        print("\n" + "="*40)
        print(" SECTION B: SPECIFIC TEST GROUPS (PAD) ")
        print("="*40)
        
        # Test 1: Tumor Type (Hamamatsu XR only)
        subset_1 = df[df['Scanner'] == 'Hamamatsu XR']
        if not subset_1.empty:
            run_pad_analysis(subset_1, 'Tumor', 'TestGroup_Tumor_in_Hamamatsu', output_subdir='TestGroups')

        # Test 2: Scanner (Breast Cancer only)
        subset_2 = df[df['Tumor'] == 'human breast cancer']
        if not subset_2.empty:
            run_pad_analysis(subset_2, 'Scanner', 'TestGroup_Scanner_in_BreastCancer', output_subdir='TestGroups')

        # Test 3: Origin (Canine Sarcoma only)
        subset_3 = df[df['Tumor'] == 'canine soft tissue sarcoma']
        if not subset_3.empty:
            run_pad_analysis(subset_3, 'Origin', 'TestGroup_Origin_in_CanineSarcoma', output_subdir='TestGroups')
            
        print("\n[DONE] All Proxy A-Distance analyses complete.")