import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

def load_features_and_domains(pkl_path='umap_and_features_with_domains.pkl'):
    """Load the processed features with domain labels"""
    with open(pkl_path, 'rb') as f:
        data = pd.read_pickle(f)
    
    # Extract features (all columns except UMAP1, UMAP2, filename, domain, domain_original)
    feature_cols = [col for col in data.columns if col not in ['UMAP1', 'UMAP2', 'filename', 'domain', 'domain_original']]
    features = data[feature_cols].values
    domains = data['domain'].values
    
    return features, domains, data

def compute_mmd(X, Y, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions
    """
    n_x, n_y = len(X), len(Y)
    
    # Compute kernel matrices
    K_xx = np.exp(-gamma * pairwise_distances(X, X, metric='euclidean')**2)
    K_yy = np.exp(-gamma * pairwise_distances(Y, Y, metric='euclidean')**2)
    K_xy = np.exp(-gamma * pairwise_distances(X, Y, metric='euclidean')**2)
    
    # MMD computation
    mmd = np.mean(K_xx) - 2 * np.mean(K_xy) + np.mean(K_yy)
    return np.sqrt(max(0, mmd))

def compute_coral(X, Y):
    """
    Compute CORAL (CORrelation ALignment) distance between two domains
    """
    # Compute covariance matrices
    cov_X = np.cov(X.T)
    cov_Y = np.cov(Y.T)
    
    # CORAL distance is the Frobenius norm of the difference
    coral_dist = np.linalg.norm(cov_X - cov_Y, 'fro')
    return coral_dist

def compute_proxy_a_distance(X, Y):
    """
    Compute Proxy A-Distance using multiple classifiers for robustness
    """
    # Create labels
    labels = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    
    # Combine data
    data = np.vstack([X, Y])
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Use multiple classifiers and take average
    classifiers = [
        LogisticRegression(C=0.1, random_state=42, max_iter=1000),
        SVC(C=0.1, kernel='linear', random_state=42, probability=True),
        LogisticRegression(C=1.0, random_state=42, max_iter=1000),
        SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, probability=True)
    ]
    
    accuracies = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for clf in classifiers:
        try:
            cv_scores = cross_val_score(
                clf, data_scaled, labels, 
                cv=skf,
                scoring='accuracy',
                n_jobs=-1
            )
            accuracies.append(np.mean(cv_scores))
        except Exception as e:
            print(f"Warning: Classifier {type(clf).__name__} failed: {e}")
            continue
    
    if not accuracies:
        print("Warning: All classifiers failed, returning default distance")
        return 1.0
    
    # Take average accuracy
    avg_accuracy = np.mean(accuracies)
    
    # Compute Proxy A-distance
    proxy_a_dist = 2 * (2 * avg_accuracy - 1)
    return np.clip(proxy_a_dist, 0, 2)

def compute_wasserstein(X, Y):
    """
    Compute average Wasserstein distance across all feature dimensions
    """
    n_features = X.shape[1]
    w_distances = []
    
    for i in range(n_features):
        w_dist = wasserstein_distance(X[:, i], Y[:, i])
        w_distances.append(w_dist)
    
    return np.mean(w_distances)

def normalize_matrix(matrix):
    """
    Normalize a distance matrix to [0, 1] range
    """
    # Get upper triangle values (excluding diagonal)
    n = matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    upper_values = matrix[upper_triangle_indices]
    
    if upper_values.size == 0 or np.max(upper_values) == np.min(upper_values):
        return matrix
    
    # Min-max normalization
    min_val = np.min(upper_values)
    max_val = np.max(upper_values)
    
    normalized = (matrix - min_val) / (max_val - min_val)
    
    # Ensure diagonal is 0
    np.fill_diagonal(normalized, 0)
    
    return normalized

def compute_domain_distances(features, domains):
    """
    Compute all four distance metrics between each pair of domains
    """
    unique_domains = sorted(np.unique(domains))
    n_domains = len(unique_domains)
    
    # Initialize distance matrices
    mmd_matrix = np.zeros((n_domains, n_domains))
    coral_matrix = np.zeros((n_domains, n_domains))
    proxy_a_matrix = np.zeros((n_domains, n_domains))
    wasserstein_matrix = np.zeros((n_domains, n_domains))
    
    # Compute pairwise distances
    for i, domain_i in enumerate(unique_domains):
        for j, domain_j in enumerate(unique_domains):
            if i < j:  # Compute only upper triangle
                # Get features for each domain
                X_i = features[domains == domain_i]
                X_j = features[domains == domain_j]
                
                # Compute distances
                mmd_matrix[i, j] = compute_mmd(X_i, X_j)
                coral_matrix[i, j] = compute_coral(X_i, X_j)
                proxy_a_matrix[i, j] = compute_proxy_a_distance(X_i, X_j)
                wasserstein_matrix[i, j] = compute_wasserstein(X_i, X_j)
                
                # Make matrices symmetric
                mmd_matrix[j, i] = mmd_matrix[i, j]
                coral_matrix[j, i] = coral_matrix[i, j]
                proxy_a_matrix[j, i] = proxy_a_matrix[i, j]
                wasserstein_matrix[j, i] = wasserstein_matrix[i, j]
    
    return {
        'MMD': mmd_matrix,
        'CORAL': coral_matrix,
        'Proxy A-Distance': proxy_a_matrix,
        'Wasserstein': wasserstein_matrix,
        'domains': unique_domains
    }

def visualize_distance_matrices(distance_results, save_path='domain_distances_visualization.png'):
    """
    Create heatmap visualizations for the four original distance matrices
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Domain Distance Matrices', fontsize=20, y=0.98)
    
    metrics = ['MMD', 'CORAL', 'Proxy A-Distance', 'Wasserstein']
    domains = distance_results['domains']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        # Create heatmap
        sns.heatmap(
            distance_results[metric],
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            square=True,
            cbar_kws={'label': f'{metric} Distance'},
            xticklabels=[f'Domain {d}' for d in domains],
            yticklabels=[f'Domain {d}' for d in domains],
            ax=ax
        )
        ax.set_title(f'{metric} Distance Matrix', fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distance matrices visualization saved to '{save_path}'")

def compute_and_visualize_weighted_average(distance_results, save_path='weighted_average_distance.png'):
    """
    Compute and visualize the normalized weighted average of all distance metrics
    """
    metrics = ['MMD', 'CORAL', 'Proxy A-Distance', 'Wasserstein']
    domains = distance_results['domains']
    
    # Normalize each matrix
    normalized_matrices = {}
    for metric in metrics:
        normalized_matrices[metric] = normalize_matrix(distance_results[metric])
    
    # Compute weighted average (equal weights of 1/4 = 0.25)
    weighted_avg = np.zeros_like(distance_results['MMD'])
    for metric in metrics:
        weighted_avg += 0.25 * normalized_matrices[metric]
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with custom colormap
    mask = np.zeros_like(weighted_avg)
    np.fill_diagonal(mask, True)
    
    sns.heatmap(
        weighted_avg,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',  # Different colormap for distinction
        square=True,
        cbar_kws={'label': 'Normalized Weighted Distance'},
        xticklabels=[f'Domain {d}' for d in domains],
        yticklabels=[f'Domain {d}' for d in domains],
        mask=mask,  # Mask diagonal
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title('Normalized Weighted Average Distance Matrix\n(25% MMD + 25% CORAL + 25% Proxy A-Distance + 25% Wasserstein)', 
              fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Weighted average distance matrix saved to '{save_path}'")
    
    # Add to results
    distance_results['Weighted_Average'] = weighted_avg
    distance_results['Normalized_Matrices'] = normalized_matrices
    
    # Print statistics for weighted average
    n_domains = len(domains)
    upper_triangle = weighted_avg[np.triu_indices(n_domains, k=1)]
    print(f"\nWeighted Average Statistics:")
    print(f"  Mean: {np.mean(upper_triangle):.4f}")
    print(f"  Std:  {np.std(upper_triangle):.4f}")
    print(f"  Min:  {np.min(upper_triangle):.4f}")
    print(f"  Max:  {np.max(upper_triangle):.4f}")
    
    return weighted_avg

def save_distance_results(distance_results, save_path='domain_distances.pkl'):
    """
    Save distance results to pickle file
    """
    with open(save_path, 'wb') as f:
        pickle.dump(distance_results, f)
    print(f"Distance results saved to '{save_path}'")

def main():
    print("Loading features and domain labels...")
    features, domains, full_data = load_features_and_domains()
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    print(f"Unique domains: {sorted(np.unique(domains))}")
    
    print("\nComputing domain distances...")
    
    distance_results = compute_domain_distances(features, domains)
    
    print("\nDistance computation complete!")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Visualize the four original distance matrices
    visualize_distance_matrices(distance_results)
    
    # Compute and visualize weighted average separately
    weighted_avg = compute_and_visualize_weighted_average(distance_results)
    
    # Save results
    save_distance_results(distance_results)
    
    # Print summary statistics for all metrics
    print("\n=== Distance Summary ===")
    metrics = ['MMD', 'CORAL', 'Proxy A-Distance', 'Wasserstein']
    for metric in metrics:
        matrix = distance_results[metric]
        n_domains = len(distance_results['domains'])
        upper_triangle = matrix[np.triu_indices(n_domains, k=1)]
        print(f"\n{metric}:")
        print(f"  Mean: {np.mean(upper_triangle):.4f}")
        print(f"  Std:  {np.std(upper_triangle):.4f}")
        print(f"  Min:  {np.min(upper_triangle):.4f}")
        print(f"  Max:  {np.max(upper_triangle):.4f}")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()