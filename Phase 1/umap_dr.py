import numpy as np
import pickle
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


def main_analysis():
    # Load features
    with open('./midog_features.pkl', 'rb') as f:
        features_dict = pickle.load(f)

    filenames = list(features_dict.keys())
    features = np.stack(list(features_dict.values()))

    # Load metadata
    metadata_df = pd.read_csv('midog.csv')
    metadata_df.columns = metadata_df.columns.str.strip()

    # Process domain labels: 1a,1b,1c -> 1; 6a,6b -> 6
    metadata_df['domain'] = metadata_df['domain'].astype(str).str.strip()
    metadata_df['domain_original'] = metadata_df['domain'].copy()
    metadata_df['domain'] = metadata_df['domain'].str.extract(r'(\d+)')[0]
    metadata_df['Slide'] = metadata_df['Slide'].astype(str).str.strip()

    # Extract slide numbers from filenames (e.g., '034.tiff' -> '34')
    slide_numbers = []
    for filename in filenames:
        base_name = os.path.splitext(filename)[0]
        try:
            slide_num = str(int(base_name))
        except ValueError:
            slide_num = base_name
        slide_numbers.append(slide_num)

    # UMAP embedding
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding = umap_model.fit_transform(features_scaled)

    # Create combined DataFrame
    features_df = pd.DataFrame(features_scaled, index=slide_numbers)
    features_df.index.name = 'Slide'

    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=slide_numbers)
    embedding_df.index.name = 'Slide'

    filename_df = pd.DataFrame({'filename': filenames}, index=slide_numbers)
    filename_df.index.name = 'Slide'

    combined_df = features_df.join(embedding_df).join(filename_df)

    # Merge with metadata
    metadata_df = metadata_df.set_index('Slide')
    final_results_df = combined_df.merge(
        metadata_df[['domain', 'domain_original']],
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Save results
    final_results_df.to_pickle('umap_and_features_with_domains.pkl')

    # Visualization
    plt.figure(figsize=(14, 10))

    unique_domains = sorted(final_results_df['domain'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
    color_dict = dict(zip(unique_domains, colors))

    for domain in unique_domains:
        mask = final_results_df['domain'] == domain
        plt.scatter(
            final_results_df.loc[mask, 'UMAP1'],
            final_results_df.loc[mask, 'UMAP2'],
            label=f'Domain {domain}',
            color=color_dict[domain],
            s=60,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )

    plt.title('UMAP Projection of Features, Colored by Domain', fontsize=16, pad=20)
    plt.xlabel('UMAP Component 1', fontsize=14)
    plt.ylabel('UMAP Component 2', fontsize=14)
    plt.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('umap_domain_visualization.png', dpi=300, bbox_inches='tight')

    print(f"Analysis complete. Processed {len(final_results_df)} slides.")
    print(f"Results saved to 'umap_and_features_with_domains.pkl'")
    print(f"Visualization saved to 'umap_domain_visualization.png'")


if __name__ == '__main__':
    plt.switch_backend('Agg')
    main_analysis()