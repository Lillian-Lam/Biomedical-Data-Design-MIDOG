import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#most of the code follows from Yang's similarity.py
METRIC_DIRS = {
    'MMD': 'mmd_unified_results',
    'CORAL': 'coral_unified_results',
    'Wasserstein': 'wasserstein_results'}

#Weights: MMD (0.5), Wass (0.3), CORAL (0.2)
WEIGHTS ={'MMD': 0.5, 'Wasserstein': 0.3, 'CORAL': 0.2}
OUTPUT_ROOT='combined_raw_shift_results'

def run_raw_combination(category, group_id, folder='Global', control_label=None):
    #Fuses raw distances from MMD, CORAL, and Wasserstein. No normalization, no similarity conversion.
    #path construction matching Yang's previous scripts
    paths={
        'MMD': f"{METRIC_DIRS['MMD']}/{folder}/MMD_{category}_{group_id}.csv",
        'CORAL': f"{METRIC_DIRS['CORAL']}/{folder}/CORAL_{category}_{group_id}.csv",
        'Wasserstein': f"{METRIC_DIRS['Wasserstein']}/{folder}/Wasserstein_{category}_{group_id}.csv"}

    raw_matrices={}
    labels, cols = None, None
    
    for metric, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            raw_matrices[metric] = df.values
            labels, cols = df.index, df.columns

    if not raw_matrices:
        print(f"Skipping {group_id}: Files not found.")
        return

    #combined=(MMD*0.5)+(Wass*0.3)+(CORAL*0.2)
    combined_raw=np.zeros_like(list(raw_matrices.values())[0])
    for metric, matrix in raw_matrices.items():
        combined_raw += matrix * WEIGHTS[metric]
    
    combined_df=pd.DataFrame(combined_raw, index=labels, columns=cols)
    save_path = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(save_path, exist_ok=True)
    combined_df.to_csv(os.path.join(save_path, f'Combined_Raw_Shift_{category}_{group_id}.csv'))

    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_df, annot=True, cmap='darkred', fmt=".2f")
    
    title_text = f"{control_label} Control" if control_label else group_id
    plt.title(f"Combined Domain Shift: {title_text}", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Heatmap_{category}_{group_id}.png"), dpi=300)
    plt.close()
    print(f"Processed: {group_id}")

if __name__ == "__main__":
    run_raw_combination('domain', 'Global_By_Domain', 'Global')
    run_raw_combination('Scanner', 'Global_By_Scanner', 'Global')

    run_raw_combination('Tumor', 'TestGroup_Tumor_in_Hamamatsu', 'TestGroups', control_label='Tumor Type')
    run_raw_combination('Scanner', 'TestGroup_Scanner_in_BreastCancer', 'TestGroups', control_label='Scanner')
    run_raw_combination('Origin', 'TestGroup_Origin_in_CanineSarcoma', 'TestGroups', control_label='Lab Origin')