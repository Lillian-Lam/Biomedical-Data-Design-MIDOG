# Phase 1 - Function Documentation

## Overview

This document provides detailed function-level documentation for all Phase 1 scripts, organized by pipeline order: Feature Extraction to Domain Shift Quantification to Similarity Score.

## Part 1: Feature Extractors

### 1.1 ctranspath_cycleGAN_norm.py - CTransPath with CycleGAN Stain Normalization

#### `has_sufficient_tissue(patch, tissue_threshold=0.1)`

**Purpose**: Check if patch contains enough tissue (not mostly white background).

**Args**:
- `patch` (np.ndarray): RGB patch array
- `tissue_threshold` (float): Minimum fraction of non-white pixels (default: 0.1)

**Returns**:
- `bool`: True if patch has sufficient tissue

#### `extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1)`

**Purpose**: Extract tissue-containing patches from WSI.

**Args**:
- `image` (PIL.Image or np.ndarray): Whole slide image
- `patch_size` (int): Size of patches in pixels (default: 224)
- `stride` (int): Step between patches in pixels (default: 224)
- `max_patches` (int): Maximum patches to extract (default: 100)
- `tissue_threshold` (float): Minimum tissue fraction (default: 0.1)

**Returns**:
- `tuple`: (patches, patch_coords) where:
  - `patches` (list): List of patch arrays, each shape (patch_size, patch_size, 3)
  - `patch_coords` (list): List of (y_start, x_start, y_end, x_end) coordinates

#### `StainNormalizer.__init__(model_weights_path, device='cuda', tile_size=256, tissue_threshold=0)`

**Purpose**: Initialize CycleGAN stain normalizer for H&E normalization.

**Args**:
- `model_weights_path` (str): Path to pretrained UnetGenerator weights
- `device` (str): 'cuda' or 'cpu'
- `tile_size` (int): Tile size for normalization in pixels (default: 256)
- `tissue_threshold` (float): Minimum tissue fraction per tile (default: 0)

#### `StainNormalizer.normalize_image(image_pil)`

**Purpose**: Normalize entire WSI using CycleGAN.

**Args**:
- `image_pil` (PIL.Image): Input WSI

**Returns**:
- `PIL.Image`: Stain-normalized image

#### `StainNormalizer._normalize_single(image_pil)`

**Purpose**: Normalize a single tile using the UnetGenerator.

**Args**:
- `image_pil` (PIL.Image): Input tile

**Returns**:
- `PIL.Image`: Normalized tile

#### `normalize_all_images(stain_normalizer, input_folder, output_folder)`

**Purpose**: Normalize all WSIs in a folder.

**Args**:
- `stain_normalizer` (StainNormalizer): Initialized normalizer
- `input_folder` (str): Path to input images
- `output_folder` (str): Path to save normalized images

**Returns**: None

**Output**: Saved TIFF files with deflate compression

#### `load_model(checkpoint_path)`

**Purpose**: Load pretrained CTransPath model.

**Args**:
- `checkpoint_path` (str): Path to .pth checkpoint file

**Returns**:
- `nn.Module`: CTransPath model with classification head removed (feature extractor only)

#### `extract_features_from_patches(model, image_folder, output_path, batch_size=32, patch_size=224, stride=224, max_patches_per_image=10000)`

**Purpose**: Extract CTransPath features from all patches in WSIs.

**Args**:
- `model` (nn.Module): CTransPath model
- `image_folder` (str): Path to WSIs (normalized or original)
- `output_path` (str): Where to save .pkl file
- `batch_size` (int): Batch size for extraction (default: 32)
- `patch_size` (int): Patch size in pixels (default: 224)
- `stride` (int): Stride in pixels (default: 224)
- `max_patches_per_image` (int): Max patches per slide (default: 10000)

**Output files**:
- `{output_path}.pkl` - Dictionary with keys: 'features', 'coordinates', 'image_size', 'num_patches'
- `{output_path}_flat.npz` - Flattened array with features, filenames, patch_ids

**Returns**:
- `dict`: Features dictionary

#### `umap_visualizations(features_path)`

**Purpose**: Generate UMAP visualizations from extracted features.

**Args**:
- `features_path` (str): Path to .pkl feature file

**Returns**:
- `pd.DataFrame`: DataFrame with UMAP coordinates and metadata
- `umap_{category}_patches.png` for categories: Tumor, Scanner, Origin, Species, Slide

### 1.2 clipnet.py - CLIP Vision Transformer

#### `load_model(model_name='ViT-B/32')`

**Purpose**: Load CLIP model and preprocessor.

**Args**:
- `model_name` (str): CLIP model variant (default: 'ViT-B/32')

**Returns**:
- `tuple`: (model, preprocess) where:
  - `model` (nn.Module): CLIP model
  - `preprocess` (Compose): CLIP-specific transforms (Resize, CenterCrop, Normalize)

#### `extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1)`

**Purpose**: Same as CTransPath version. Extracts tissue-containing patches.

**Returns**: `tuple` (patches, patch_coords)

#### `extract_features_from_patches(model, preprocess, image_folder, output_path, batch_size=32, patch_size=224, stride=224, max_patches_per_image=100)`

**Purpose**: Extract CLIP features from patches.

**Args**:
- `model` (nn.Module): CLIP model
- `preprocess` (Compose): CLIP preprocessing function
- Same other args as CTransPath

**Returns**: `dict` - Features dictionary

**Feature dimension**: 512 for ViT-B/32

### 1.3 dinov3.py - DINOv3 Vision Transformer

#### `get_dinov3_transform(patch_size=224)`

**Purpose**: Create ImageNet normalization transform for DINOv3.

**Args**:
- `patch_size` (int): Resize target size (default: 224)

**Returns**:
- `transforms.Compose`: Resize -> ToTensor -> Normalize

#### `extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1)`

**Purpose**: Same as above.

**Returns**: `tuple` (patches, patch_coords)

#### `load_model(model_name='vit_base_patch16_dinov3.lvd1689m')`

**Purpose**: Load DINOv3 model via timm.

**Args**:
- `model_name` (str): timm model identifier

**Returns**:
- `tuple`: (model, preprocess) where model outputs normalized features

**Requirements**: timm >= 1.0.20

**Feature dimension**: 768 for ViT-Base

#### `extract_features_from_patches(model, preprocess, image_folder, output_path, batch_size=32, patch_size=224, stride=224, max_patches_per_image=100)`

**Purpose**: Extract DINOv3 features from patches.

**Returns**: `dict` - Features dictionary

### 1.4 efficientnet.py - EfficientNet CNN

#### `extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1)`

**Purpose**: Same as above.

**Returns**: `tuple` (patches, patch_coords)

#### `load_model(model_name='efficientnet_b0', pretrained=True)`

**Purpose**: Load EfficientNet model.

**Args**:
- `model_name` (str): One of 'efficientnet_b0', 'b1', 'b2', 'b3', 'b4'
- `pretrained` (bool): Use ImageNet weights (default: True)

**Returns**:
- `nn.Module`: EfficientNet with classifier head removed (feature extractor)

#### `extract_features_from_patches(model, image_folder, output_path, batch_size=32, patch_size=224, stride=224, max_patches_per_image=100)`

**Purpose**: Extract EfficientNet features from patches.

**Returns**: `dict` - Features dictionary

### 1.5 mae.py - Masked Autoencoder (MAE)

#### `get_mae_transform(patch_size=224)`

**Purpose**: Create ImageNet normalization for MAE.

**Returns**: `transforms.Compose`: Resize -> ToTensor -> Normalize

#### `extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1)`

**Purpose**: Same as above.

**Returns**: `tuple` (patches, patch_coords)

#### `load_model(model_name='mae_vit_base_patch16')`

**Purpose**: Load MAE ViT-Base model.

**Returns**:
- `tuple`: (model, preprocess) where:
  - `model` has `is_timm` attribute indicating source
  - If timm: uses `forward_features()` and takes CLS token
  - If official: uses `forward_encoder()` with mask_ratio=0, takes CLS token

**Feature dimension**: 768

#### `extract_features_from_patches(model, preprocess, image_folder, output_path, batch_size=32, patch_size=224, stride=224, max_patches_per_image=100)`

**Purpose**: Extract MAE features from patches.

**Returns**: `dict` - Features dictionary

### 1.6 UMAP Visualization (for each encoder)

Each feature extractor has a corresponding `umap_{encoder}.py` script.

#### `umap_visualizations()`

**Purpose**: Load features, reduce dimensionality with UMAP, and generate colored plots.

**Input files** (expects):
- `./midog_{encoder}_features_patches.pkl` - Feature dictionary
- `midog.csv` - Metadata

**Returns**:
- `pd.DataFrame`: DataFrame with UMAP coordinates and metadata

**Output files**:
- `{encoder}_umap_{category}_patches.png` - UMAP plots for categories:
  - Slide, Tumor, Scanner, Origin, Species
- `umap_results_with_metadata_{encoder}.csv` - Full results

**UMAP parameters**:

| Parameter | Value |
|-----------|-------|
| `n_neighbors` | 15 |
| `n_components` | 2 |
| `min_dist` | 0.1 |
| `metric` | 'cosine' |
| `random_state` | 42 |

## Part 2: Domain Shift Quantification

### 2.1 MMD_v1.py - Maximum Mean Discrepancy

#### `mmd_unbiased(X, Y, use_scaled=True)`

**Purpose**: Computes unbiased MMD estimator using linear kernel with O(N) memory optimization.

**Args**:
- `X` (np.ndarray): Source domain features, shape (n_samples, n_features)
- `Y` (np.ndarray): Target domain features, shape (m_samples, n_features)
- `use_scaled` (bool): If True, scales features by pooled standard deviation (default: True)

**Returns**:
- `float`: Squared MMD distance between domains

**Mathematical formula**: 
```
MMD^2 = (sum_XX_off_diag)/(n(n-1)) + (sum_YY_off_diag)/(m(m-1)) - 2*(sum_XY)/(n*m)
```

#### `mmd_intra_domain(X, n_splits=5, use_scaled=True)`

**Purpose**: Calculates intra-domain variation (diagonal of distance matrix) by splitting data into random halves.

**Args**:
- `X` (np.ndarray): Features from single domain
- `n_splits` (int): Number of random splits to average over (default: 5)
- `use_scaled` (bool): Passed to `mmd_unbiased`

**Returns**:
- `float`: Average MMD between random halves (expected close to 0 if domain is homogeneous)

#### `load_and_prepare_dataframe(pkl_path, csv_path)`

**Purpose**: Loads pickle features and CSV metadata, merging into pandas DataFrame.

**Args**:
- `pkl_path` (str): Path to .pkl file with features (output from feature extractors)
- `csv_path` (str): Path to midog.csv metadata file

**Returns**:
- `pd.DataFrame` or `None`: DataFrame with columns: filename, slide_id, domain, Scanner, Tumor, Origin, Species, features

**Expected CSV columns**: Slide, Scanner, Tumor, Origin, Species, Dataset

**Data cleaning**: 
- Strips whitespace from column names
- Fixes 'Hamammatsu XR' -> 'Hamamatsu XR'
- Filters to Dataset == 'train' if column exists

#### `run_mmd_analysis(df, group_col, title_suffix, output_subdir='')`

**Purpose**: Main function to compute MMD matrix, generate heatmap, and print top 5 closest relations.

**Args**:
- `df` (pd.DataFrame): DataFrame from `load_and_prepare_dataframe`
- `group_col` (str): Column name to group by ('domain', 'Scanner', 'Tumor', 'Origin')
- `title_suffix` (str): Suffix for output filenames
- `output_subdir` (str): Subdirectory in `mmd_unified_results/` (default: '')

**Returns**: None

**Output files**:
- `MMD_{group_col}_{title_suffix}.csv` - Distance matrix
- `MMD_{group_col}_{title_suffix}.png` - Heatmap 

**Console output**: Top 5 closest domain pairs (lowest MMD)

### 2.2 CORAL_v1.py - Correlation Alignment

#### `compute_covariance(features)`

**Purpose**: Computes covariance matrix for feature set.

**Args**:
- `features` (np.ndarray): Shape (n_samples, n_features)

**Returns**:
- `np.ndarray`: Covariance matrix, shape (n_features, n_features)

**Note**: Returns zero matrix if n_samples < 2

#### `coral_distance(features_a, features_b)`

**Purpose**: Calculates CORAL distance between two domains.

**Args**:
- `features_a` (np.ndarray): Source domain features
- `features_b` (np.ndarray): Target domain features

**Returns**:
- `float`: Squared Frobenius norm of covariance difference

**Formula**: `||Cov(A) - Cov(B)||_F^2`

**Interpretation**: Captures 'style' or 'texture' second-order statistics

#### `calculate_intra_domain_variation(features, n_splits=5)`

**Purpose**: Calculates within-domain variation (diagonal) via random splits.

**Args**:
- `features` (np.ndarray): Features from single domain
- `n_splits` (int): Number of random splits (default: 5)

**Returns**: `float` - Average CORAL distance between halves

#### `load_and_prepare_dataframe(pkl_path, csv_path)`

**Purpose**: Same as MMD_v1 version.

**Returns**: `pd.DataFrame` or `None`

#### `run_analysis(df, group_col, title_suffix, output_subdir='')`

**Purpose**: Main function to compute CORAL matrix, heatmap, and ranking.

**Args**:
- Same as `run_mmd_analysis`

**Returns**: None

**Output files**:
- `CORAL_{group_col}_{title_suffix}.csv`
- `CORAL_{group_col}_{title_suffix}.png`

**Console output**: Top 5 strongest relations (lowest distance)

### 2.3 Wasserstein Distance_v1.py

#### `compute_wasserstein_distance(features_a, features_b)`

**Purpose**: Calculates average marginal Wasserstein distance (Earth Mover's Distance).

**Args**:
- `features_a` (np.ndarray): Features shape (n, d)
- `features_b` (np.ndarray): Features shape (m, d)

**Returns**:
- `float`: Average Wasserstein distance across all feature dimensions

**Process**: For each dimension, compute 1D Wasserstein using `scipy.stats.wasserstein_distance`, then average.

**Interpretation**: Captures geometric discrepancy between distributions

#### `calculate_intra_domain_variation(features, n_splits=5)`

**Purpose**: Calculates within-domain variation.

**Args**:
- `features` (np.ndarray): Feature matrix
- `n_splits` (int): Number of random splits (default: 5)

**Returns**: `float` - Average Wasserstein distance between halves

#### `load_and_prepare_dataframe(pkl_path, csv_path)`

**Purpose**: Same as above.

**Returns**: `pd.DataFrame` or `None`

#### `run_analysis(df, group_col, title_suffix, output_subdir='')`

**Purpose**: Main function to compute Wasserstein matrix, heatmap, and ranking.

**Returns**: None

**Output files**:
- `Wasserstein_{group_col}_{title_suffix}.csv`
- `Wasserstein_{group_col}_{title_suffix}.png`

### 2.4 Proxy A-Distance.py

#### `compute_proxy_a_distance(features_source, features_target, cv_folds=5)`

**Purpose**: Computes Proxy A-Distance (PAD) using Linear SVM classifier.

**Args**:
- `features_source` (np.ndarray): Source domain features
- `features_target` (np.ndarray): Target domain features
- `cv_folds` (int): Number of cross-validation folds (default: 5)

**Returns**:
- `float`: PAD value in [0, 2]
  
**Formula**: `PAD = 2 * (1 - 2 * error) = 2 * (2 * accuracy - 1)`

**Classifier**: LinearSVC with StandardScaler in pipeline

#### `calculate_intra_domain_pad(features, cv_folds=5)`

**Purpose**: Calculates self-PAD (diagonal) by splitting domain into halves.

**Args**:
- `features` (np.ndarray): Feature matrix
- `cv_folds` (int): Number of CV folds

**Returns**: `float` - Expected to be close to 0 (indistinguishable from self)

#### `load_and_prepare_dataframe(pkl_path, csv_path)`

**Purpose**: Same as above.

**Returns**: `pd.DataFrame` or `None`

#### `run_pad_analysis(df, group_col, title_suffix, output_subdir='')`

**Purpose**: Main function to compute PAD matrix, heatmap, and ranking.

**Returns**: None

**Output files**:
- `PAD_{group_col}_{title_suffix}.csv`
- `PAD_{group_col}_{title_suffix}.png`

**Console output**: Top 5 largest domain shifts (highest PAD)

## Part 3: Similarity

### 3.1 domain_shift_score.py - Raw Fusion

#### `run_raw_combination(category, group_id, folder='Global', control_label=None)`

**Purpose**: Fuse raw distances from MMD, CORAL, and Wasserstein using weighted sum (no normalization, no similarity conversion).

**Args**:
- `category` (str): Category name ('domain', 'Scanner', 'Tumor', 'Origin')
- `group_id` (str): Specific group identifier (e.g., 'Global_By_Domain')
- `folder` (str): 'Global' or 'TestGroups' (default: 'Global')
- `control_label` (str): Optional label for plot title

**Returns**: None

**Weights**:

| Metric | Weight |
|--------|--------|
| MMD | 0.5 |
| Wasserstein | 0.3 |
| CORAL | 0.2 |

**Output files**:
- `Combined_Raw_Shift_{category}_{group_id}.csv` - Fused distance matrix
- `Heatmap_{category}_{group_id}.png` - Heatmap visualization

### 3.2 similarity.py - Domain-Level Fusion

#### `load_matrix(csv_path)`

**Purpose**: Load symmetric distance matrix from CSV.

**Args**:
- `csv_path` (str): Path to CSV file (first column as index)

**Returns**: `pd.DataFrame` or `None`

#### `normalize_matrix(df)`

**Purpose**: Min-Max normalize matrix to [0, 1] range.

**Args**:
- `df` (pd.DataFrame): Distance matrix

**Returns**: `pd.DataFrame` - Normalized matrix

**Formula**: `(value - min) / (max - min)`

#### `distance_to_similarity(df, sigma=0.5)`

**Purpose**: Convert distance matrix to similarity using Gaussian RBF kernel.

**Args**:
- `df` (pd.DataFrame): Normalized distance matrix
- `sigma` (float): Kernel bandwidth temperature (default: 0.5)

**Returns**: `pd.DataFrame` - Similarity matrix

**Formula**: `sim = exp(-distance^2 / (2 * sigma^2))`

**Interpretation**:
- distance = 0 -> similarity = 1.0
- distance large -> similarity -> 0.0

**Output files** (from main execution):
- `Final_Similarity_Matrix.csv` - Fused similarity matrix
- `Final_Similarity_Matrix.png` - Heatmap

**Default weights**: MMD=0.4, Wasserstein=0.4, CORAL=0.2

### 3.3 similarity_wsi.py - WSI-Level Fusion

#### `mmd_linear(X, Y)`

**Purpose**: Memory-efficient MMD with linear kernel (O(N) complexity).

**Args**:
- `X` (np.ndarray): Features from slide A, shape (n, d)
- `Y` (np.ndarray): Features from slide B, shape (m, d)

**Returns**: `float` - MMD² distance

#### `coral_dist(X, Y)`

**Purpose**: CORAL distance between two slides.

**Args**:
- `X` (np.ndarray): Slide A features
- `Y` (np.ndarray): Slide B features

**Returns**: `float` - Squared Frobenius norm of covariance difference

#### `wasserstein_marginal(X, Y)`

**Purpose**: Marginal Wasserstein distance averaged across dimensions.

**Args**:
- `X` (np.ndarray): Slide A features
- `Y` (np.ndarray): Slide B features

**Returns**: `float` - Average 1D Wasserstein distance

#### `normalize_matrix(mat)`

**Purpose**: Global min-max normalization.

**Args**:
- `mat` (np.ndarray): Distance matrix

**Returns**: `np.ndarray` - Normalized matrix to [0, 1]

#### `dist_to_sim(mat, sigma=0.5)`

**Purpose**: Gaussian kernel similarity conversion.

**Args**:
- `mat` (np.ndarray): Distance matrix
- `sigma` (float): Kernel bandwidth

**Returns**: `np.ndarray` - Similarity matrix

#### `load_data(pkl, csv)`

**Purpose**: Load features and group by slide ID.

**Args**:
- `pkl` (str): Path to .pkl feature file
- `csv` (str): Path to midog.csv

**Returns**:
- `dict`: Keys = slide IDs (as strings), Values = stacked patch features (np.ndarray)

**Constants**:
- `MAX_PATCHES_PER_SLIDE = 2000` (downsampling for computational efficiency)

#### `generate_fused_matrix(slide_feats)`

**Purpose**: Compute all three metrics, normalize, convert to similarity, and fuse.

**Args**:
- `slide_feats` (dict): From `load_data`

**Returns**:
- `tuple`: (final_similarity_matrix, sorted_slide_ids)

**Process**:
1. Compute MMD, CORAL, Wasserstein for all slide pairs
2. Global min-max normalization per metric
3. Convert distance -> similarity (Gaussian kernel, σ=0.5)
4. Weighted fusion: MMD=0.5, Wasserstein=0.3, CORAL=0.2
5. Set diagonal to 1.0 (perfect self-similarity)

**Output files**:
- `WSI_Fused_Similarity_Matrix.csv` - Fused similarity matrix for all WSIs
- `WSI_Fused_Heatmap.png` - Heatmap with axis ticks hidden for large N

Gemini was used to help format and draft the documentation.md based on my original code. I reviewed and edited all descriptions for technical accuracy.





