

\# Phase 1 - Function Documentation



\## Overview

This document provides detailed function-level documentation for all Phase 1 scripts, organized by pipeline order: Feature Extraction to Domain Shift Quantification to Similarity Score.





\## Part 1: Feature Extractors



\### 1.1 ctranspath\_cycleGAN\_norm.py - CTransPath with CycleGAN Stain Normalization



\#### `has\_sufficient\_tissue(patch, tissue\_threshold=0.1)`

\*\*Purpose\*\*: Check if patch contains enough tissue (not mostly white background).



\*\*Args\*\*:

\- `patch` (np.ndarray): RGB patch array

\- `tissue\_threshold` (float): Minimum fraction of non-white pixels (default: 0.1)



\*\*Returns\*\*:

\- `bool`: True if patch has sufficient tissue



\#### `extract\_tissue\_patches(image, patch\_size=224, stride=224, max\_patches=100, tissue\_threshold=0.1)`

\*\*Purpose\*\*: Extract tissue-containing patches from WSI.



\*\*Args\*\*:

\- `image` (PIL.Image or np.ndarray): Whole slide image

\- `patch\_size` (int): Size of patches in pixels (default: 224)

\- `stride` (int): Step between patches in pixels (default: 224)

\- `max\_patches` (int): Maximum patches to extract (default: 100)

\- `tissue\_threshold` (float): Minimum tissue fraction (default: 0.1)



\*\*Returns\*\*:

\- `tuple`: (patches, patch\_coords) where:

&#x20; - `patches` (list): List of patch arrays, each shape (patch\_size, patch\_size, 3)

&#x20; - `patch\_coords` (list): List of (y\_start, x\_start, y\_end, x\_end) coordinates



\#### `StainNormalizer.\_\_init\_\_(model\_weights\_path, device='cuda', tile\_size=256, tissue\_threshold=0)`

\*\*Purpose\*\*: Initialize CycleGAN stain normalizer for H\&E normalization.



\*\*Args\*\*:

\- `model\_weights\_path` (str): Path to pretrained UnetGenerator weights

\- `device` (str): 'cuda' or 'cpu'

\- `tile\_size` (int): Tile size for normalization in pixels (default: 256)

\- `tissue\_threshold` (float): Minimum tissue fraction per tile (default: 0)



\#### `StainNormalizer.normalize\_image(image\_pil)`

\*\*Purpose\*\*: Normalize entire WSI using CycleGAN.



\*\*Args\*\*:

\- `image\_pil` (PIL.Image): Input WSI



\*\*Returns\*\*:

\- `PIL.Image`: Stain-normalized image



\#### `StainNormalizer.\_normalize\_single(image\_pil)`

\*\*Purpose\*\*: Normalize a single tile using the UnetGenerator.



\*\*Args\*\*:

\- `image\_pil` (PIL.Image): Input tile



\*\*Returns\*\*:

\- `PIL.Image`: Normalized tile



\#### `normalize\_all\_images(stain\_normalizer, input\_folder, output\_folder)`

\*\*Purpose\*\*: Normalize all WSIs in a folder.



\*\*Args\*\*:

\- `stain\_normalizer` (StainNormalizer): Initialized normalizer

\- `input\_folder` (str): Path to input images

\- `output\_folder` (str): Path to save normalized images



\*\*Returns\*\*: None



\*\*Output\*\*: Saved TIFF files with deflate compression



\#### `load\_model(checkpoint\_path)`

\*\*Purpose\*\*: Load pretrained CTransPath model.



\*\*Args\*\*:

\- `checkpoint\_path` (str): Path to .pth checkpoint file



\*\*Returns\*\*:

\- `nn.Module`: CTransPath model with classification head removed (feature extractor only)



\#### `extract\_features\_from\_patches(model, image\_folder, output\_path, batch\_size=32, patch\_size=224, stride=224, max\_patches\_per\_image=10000)`

\*\*Purpose\*\*: Extract CTransPath features from all patches in WSIs.



\*\*Args\*\*:

\- `model` (nn.Module): CTransPath model

\- `image\_folder` (str): Path to WSIs (normalized or original)

\- `output\_path` (str): Where to save .pkl file

\- `batch\_size` (int): Batch size for extraction (default: 32)

\- `patch\_size` (int): Patch size in pixels (default: 224)

\- `stride` (int): Stride in pixels (default: 224)

\- `max\_patches\_per\_image` (int): Max patches per slide (default: 10000)



\*\*Output files\*\*:

\- `{output\_path}.pkl` - Dictionary with keys: 'features', 'coordinates', 'image\_size', 'num\_patches'

\- `{output\_path}\_flat.npz` - Flattened array with features, filenames, patch\_ids



\*\*Returns\*\*:

\- `dict`: Features dictionary



\#### `umap\_visualizations(features\_path)`

\*\*Purpose\*\*: Generate UMAP visualizations from extracted features.



\*\*Args\*\*:

\- `features\_path` (str): Path to .pkl feature file



\*\*Returns\*\*:

\- `pd.DataFrame`: DataFrame with UMAP coordinates and metadata

\- `umap\_{category}\_patches.png` for categories: Tumor, Scanner, Origin, Species, Slide



\### 1.2 clipnet.py - CLIP Vision Transformer



\#### `load\_model(model\_name='ViT-B/32')`

\*\*Purpose\*\*: Load CLIP model and preprocessor.



\*\*Args\*\*:

\- `model\_name` (str): CLIP model variant (default: 'ViT-B/32')



\*\*Returns\*\*:

\- `tuple`: (model, preprocess) where:

&#x20; - `model` (nn.Module): CLIP model

&#x20; - `preprocess` (Compose): CLIP-specific transforms (Resize, CenterCrop, Normalize)



\#### `extract\_tissue\_patches(image, patch\_size=224, stride=224, max\_patches=100, tissue\_threshold=0.1)`

\*\*Purpose\*\*: Same as CTransPath version. Extracts tissue-containing patches.



\*\*Returns\*\*: `tuple` (patches, patch\_coords)



\#### `extract\_features\_from\_patches(model, preprocess, image\_folder, output\_path, batch\_size=32, patch\_size=224, stride=224, max\_patches\_per\_image=100)`

\*\*Purpose\*\*: Extract CLIP features from patches.



\*\*Args\*\*:

\- `model` (nn.Module): CLIP model

\- `preprocess` (Compose): CLIP preprocessing function

\- Same other args as CTransPath



\*\*Returns\*\*: `dict` - Features dictionary



\*\*Feature dimension\*\*: 512 for ViT-B/32



\### 1.3 dinov3.py - DINOv3 Vision Transformer



\#### `get\_dinov3\_transform(patch\_size=224)`

\*\*Purpose\*\*: Create ImageNet normalization transform for DINOv3.



\*\*Args\*\*:

\- `patch\_size` (int): Resize target size (default: 224)



\*\*Returns\*\*:

\- `transforms.Compose`: Resize ->ToTensor ->Normalize



\#### `extract\_tissue\_patches(image, patch\_size=224, stride=224, max\_patches=100, tissue\_threshold=0.1)`

\*\*Purpose\*\*: Same as above.



\*\*Returns\*\*: `tuple` (patches, patch\_coords)



\#### `load\_model(model\_name='vit\_base\_patch16\_dinov3.lvd1689m')`

\*\*Purpose\*\*: Load DINOv3 model via timm.



\*\*Args\*\*:

\- `model\_name` (str): timm model identifier



\*\*Returns\*\*:

\- `tuple`: (model, preprocess) where model outputs normalized features



\*\*Requirements\*\*: timm >= 1.0.20



\*\*Feature dimension\*\*: 768 for ViT-Base



\#### `extract\_features\_from\_patches(model, preprocess, image\_folder, output\_path, batch\_size=32, patch\_size=224, stride=224, max\_patches\_per\_image=100)`

\*\*Purpose\*\*: Extract DINOv3 features from patches.



\*\*Returns\*\*: `dict` - Features dictionary



\### 1.4 efficientnet.py - EfficientNet CNN



\#### `extract\_tissue\_patches(image, patch\_size=224, stride=224, max\_patches=100, tissue\_threshold=0.1)`

\*\*Purpose\*\*: Same as above.



\*\*Returns\*\*: `tuple` (patches, patch\_coords)



\#### `load\_model(model\_name='efficientnet\_b0', pretrained=True)`

\*\*Purpose\*\*: Load EfficientNet model.



\*\*Args\*\*:

\- `model\_name` (str): One of 'efficientnet\_b0', 'b1', 'b2', 'b3', 'b4'

\- `pretrained` (bool): Use ImageNet weights (default: True)



\*\*Returns\*\*:

\- `nn.Module`: EfficientNet with classifier head removed (feature extractor)



\#### `extract\_features\_from\_patches(model, image\_folder, output\_path, batch\_size=32, patch\_size=224, stride=224, max\_patches\_per\_image=100)`

\*\*Purpose\*\*: Extract EfficientNet features from patches.



\*\*Returns\*\*: `dict` - Features dictionary



\### 1.5 mae.py - Masked Autoencoder (MAE)



\#### `get\_mae\_transform(patch\_size=224)`

\*\*Purpose\*\*: Create ImageNet normalization for MAE.



\*\*Returns\*\*: `transforms.Compose`: Resize ->ToTensor ->Normalize



\#### `extract\_tissue\_patches(image, patch\_size=224, stride=224, max\_patches=100, tissue\_threshold=0.1)`

\*\*Purpose\*\*: Same as above.



\*\*Returns\*\*: `tuple` (patches, patch\_coords)



\#### `load\_model(model\_name='mae\_vit\_base\_patch16')`

\*\*Purpose\*\*: Load MAE ViT-Base model.



\*\*Returns\*\*:

\- `tuple`: (model, preprocess) where:

&#x20; - `model` has `is\_timm` attribute indicating source

&#x20; - If timm: uses `forward\_features()` and takes CLS token

&#x20; - If official: uses `forward\_encoder()` with mask\_ratio=0, takes CLS token



\*\*Feature dimension\*\*: 768



\#### `extract\_features\_from\_patches(model, preprocess, image\_folder, output\_path, batch\_size=32, patch\_size=224, stride=224, max\_patches\_per\_image=100)`

\*\*Purpose\*\*: Extract MAE features from patches.



\*\*Returns\*\*: `dict` - Features dictionary



\### 1.6 UMAP Visualization (for each encoder)



Each feature extractor has a corresponding `umap\_{encoder}.py` script.



\#### `umap\_visualizations()`

\*\*Purpose\*\*: Load features, reduce dimensionality with UMAP, and generate colored plots.



\*\*Input files\*\* (expects):

\- `./midog\_{encoder}\_features\_patches.pkl` - Feature dictionary

\- `midog.csv` - Metadata



\*\*Returns\*\*:

\- `pd.DataFrame`: DataFrame with UMAP coordinates and metadata



\*\*Output files\*\*:

\- `{encoder}\_umap\_{category}\_patches.png` - UMAP plots for categories:

&#x20; - Slide, Tumor, Scanner, Origin, Species

\- `umap\_results\_with\_metadata\_{encoder}.csv` - Full results



\*\*UMAP parameters\*\*:

| Parameter | Value |

|-----------|-------|

| `n\_neighbors` | 15 |

| `n\_components` | 2 |

| `min\_dist` | 0.1 |

| `metric` | 'cosine' |

| `random\_state` | 42 |



\## Part 2: Domain Shift Quantification



\### 2.1 MMD\_v1.py - Maximum Mean Discrepancy



\#### `mmd\_unbiased(X, Y, use\_scaled=True)`

\*\*Purpose\*\*: Computes unbiased MMD estimator using linear kernel with O(N) memory optimization.



\*\*Args\*\*:

\- `X` (np.ndarray): Source domain features, shape (n\_samples, n\_features)

\- `Y` (np.ndarray): Target domain features, shape (m\_samples, n\_features)

\- `use\_scaled` (bool): If True, scales features by pooled standard deviation (default: True)



\*\*Returns\*\*:

\- `float`: Squared MMD distance between domains



\*\*Mathematical formula\*\*: 

```

MMD^2 = (sum\_XX\_off\_diag)/(n(n-1)) + (sum\_YY\_off\_diag)/(m(m-1)) - 2\*(sum\_XY)/(n\*m)

```



\#### `mmd\_intra\_domain(X, n\_splits=5, use\_scaled=True)`

\*\*Purpose\*\*: Calculates intra-domain variation (diagonal of distance matrix) by splitting data into random halves.



\*\*Args\*\*:

\- `X` (np.ndarray): Features from single domain

\- `n\_splits` (int): Number of random splits to average over (default: 5)

\- `use\_scaled` (bool): Passed to `mmd\_unbiased`



\*\*Returns\*\*:

\- `float`: Average MMD between random halves (expected close to 0 if domain is homogeneous)



\#### `load\_and\_prepare\_dataframe(pkl\_path, csv\_path)`

\*\*Purpose\*\*: Loads pickle features and CSV metadata, merging into pandas DataFrame.



\*\*Args\*\*:

\- `pkl\_path` (str): Path to .pkl file with features (output from feature extractors)

\- `csv\_path` (str): Path to midog.csv metadata file



\*\*Returns\*\*:

\- `pd.DataFrame` or `None`: DataFrame with columns: filename, slide\_id, domain, Scanner, Tumor, Origin, Species, features



\*\*Expected CSV columns\*\*: Slide, Scanner, Tumor, Origin, Species, Dataset



\*\*Data cleaning\*\*: 

\- Strips whitespace from column names

\- Fixes 'Hamammatsu XR' ->'Hamamatsu XR'

\- Filters to Dataset == 'train' if column exists



\#### `run\_mmd\_analysis(df, group\_col, title\_suffix, output\_subdir='')`

\*\*Purpose\*\*: Main function to compute MMD matrix, generate heatmap, and print top 5 closest relations.



\*\*Args\*\*:

\- `df` (pd.DataFrame): DataFrame from `load\_and\_prepare\_dataframe`

\- `group\_col` (str): Column name to group by ('domain', 'Scanner', 'Tumor', 'Origin')

\- `title\_suffix` (str): Suffix for output filenames

\- `output\_subdir` (str): Subdirectory in `mmd\_unified\_results/` (default: '')



\*\*Returns\*\*: None



\*\*Output files\*\*:

\- `MMD\_{group\_col}\_{title\_suffix}.csv` - Distance matrix

\-`MMD\_{group\_col}\_{title\_suffix}.png` - Heatmap 



\*\*Console output\*\*: Top 5 closest domain pairs (lowest MMD)



\### 2.2 CORAL\_v1.py - Correlation Alignment



\#### `compute\_covariance(features)`

\*\*Purpose\*\*: Computes covariance matrix for feature set.



\*\*Args\*\*:

\- `features` (np.ndarray): Shape (n\_samples, n\_features)



\*\*Returns\*\*:

\- `np.ndarray`: Covariance matrix, shape (n\_features, n\_features)



\*\*Note\*\*: Returns zero matrix if n\_samples < 2



\#### `coral\_distance(features\_a, features\_b)`

\*\*Purpose\*\*: Calculates CORAL distance between two domains.



\*\*Args\*\*:

\- `features\_a` (np.ndarray): Source domain features

\- `features\_b` (np.ndarray): Target domain features



\*\*Returns\*\*:

\- `float`: Squared Frobenius norm of covariance difference



\*\*Formula\*\*: `||Cov(A) - Cov(B)||\_F^2`



\*\*Interpretation\*\*: Captures 'style' or 'texture' second-order statistics



\#### `calculate\_intra\_domain\_variation(features, n\_splits=5)`

\*\*Purpose\*\*: Calculates within-domain variation (diagonal) via random splits.



\*\*Args\*\*:

\- `features` (np.ndarray): Features from single domain

\- `n\_splits` (int): Number of random splits (default: 5)



\*\*Returns\*\*: `float` - Average CORAL distance between halves



\#### `load\_and\_prepare\_dataframe(pkl\_path, csv\_path)`

\*\*Purpose\*\*: Same as MMD\_v1 version.



\*\*Returns\*\*: `pd.DataFrame` or `None`



\#### `run\_analysis(df, group\_col, title\_suffix, output\_subdir='')`

\*\*Purpose\*\*: Main function to compute CORAL matrix, heatmap, and ranking.



\*\*Args\*\*:

\- Same as `run\_mmd\_analysis`



\*\*Returns\*\*: None



\*\*Output files\*\*:

\- `CORAL\_{group\_col}\_{title\_suffix}.csv`

\- `CORAL\_{group\_col}\_{title\_suffix}.png`



\*\*Console output\*\*: Top 5 strongest relations (lowest distance)



\### 2.3 Wasserstein Distance\_v1.py



\#### `compute\_wasserstein\_distance(features\_a, features\_b)`

\*\*Purpose\*\*: Calculates average marginal Wasserstein distance (Earth Mover's Distance).



\*\*Args\*\*:

\- `features\_a` (np.ndarray): Features shape (n, d)

\- `features\_b` (np.ndarray): Features shape (m, d)



\*\*Returns\*\*:

\- `float`: Average Wasserstein distance across all feature dimensions



\*\*Process\*\*: For each dimension, compute 1D Wasserstein using `scipy.stats.wasserstein\_distance`, then average.



\*\*Interpretation\*\*: Captures geometric discrepancy between distributions



\#### `calculate\_intra\_domain\_variation(features, n\_splits=5)`

\*\*Purpose\*\*: Calculates within-domain variation.



\*\*Args\*\*:

\- `features` (np.ndarray): Feature matrix

\- `n\_splits` (int): Number of random splits (default: 5)



\*\*Returns\*\*: `float` - Average Wasserstein distance between halves



\#### `load\_and\_prepare\_dataframe(pkl\_path, csv\_path)`

\*\*Purpose\*\*: Same as above.



\*\*Returns\*\*: `pd.DataFrame` or `None`



\#### `run\_analysis(df, group\_col, title\_suffix, output\_subdir='')`

\*\*Purpose\*\*: Main function to compute Wasserstein matrix, heatmap, and ranking.



\*\*Returns\*\*: None



\*\*Output files\*\*:

\- `Wasserstein\_{group\_col}\_{title\_suffix}.csv`

\- `Wasserstein\_{group\_col}\_{title\_suffix}.png`



\### 2.4 Proxy A-Distance.py



\#### `compute\_proxy\_a\_distance(features\_source, features\_target, cv\_folds=5)`

\*\*Purpose\*\*: Computes Proxy A-Distance (PAD) using Linear SVM classifier.



\*\*Args\*\*:

\- `features\_source` (np.ndarray): Source domain features

\- `features\_target` (np.ndarray): Target domain features

\- `cv\_folds` (int): Number of cross-validation folds (default: 5)



\*\*Returns\*\*:

\- `float`: PAD value in \[0, 2]

&#x20; 

\*\*Formula\*\*: `PAD = 2 \* (1 - 2 \* error) = 2 \* (2 \* accuracy - 1)`



\*\*Classifier\*\*: LinearSVC with StandardScaler in pipeline



\#### `calculate\_intra\_domain\_pad(features, cv\_folds=5)`

\*\*Purpose\*\*: Calculates self-PAD (diagonal) by splitting domain into halves.



\*\*Args\*\*:

\- `features` (np.ndarray): Feature matrix

\- `cv\_folds` (int): Number of CV folds



\*\*Returns\*\*: `float` - Expected to be close to 0 (indistinguishable from self)





\#### `load\_and\_prepare\_dataframe(pkl\_path, csv\_path)`

\*\*Purpose\*\*: Same as above.



\*\*Returns\*\*: `pd.DataFrame` or `None`



\#### `run\_pad\_analysis(df, group\_col, title\_suffix, output\_subdir='')`

\*\*Purpose\*\*: Main function to compute PAD matrix, heatmap, and ranking.



\*\*Returns\*\*: None



\*\*Output files\*\*:

\- `PAD\_{group\_col}\_{title\_suffix}.csv`

\- `PAD\_{group\_col}\_{title\_suffix}.png`



\*\*Console output\*\*: Top 5 largest domain shifts (highest PAD)



\## Part 3: Similarity



\### 3.1 domain\_shift\_score.py - Raw Fusion



\#### `run\_raw\_combination(category, group\_id, folder='Global', control\_label=None)`

\*\*Purpose\*\*: Fuse raw distances from MMD, CORAL, and Wasserstein using weighted sum (no normalization, no similarity conversion).



\*\*Args\*\*:

\- `category` (str): Category name ('domain', 'Scanner', 'Tumor', 'Origin')

\- `group\_id` (str): Specific group identifier (e.g., 'Global\_By\_Domain')

\- `folder` (str): 'Global' or 'TestGroups' (default: 'Global')

\- `control\_label` (str): Optional label for plot title



\*\*Returns\*\*: None



\*\*Weights\*\*:

| Metric | Weight |

|--------|--------|

| MMD | 0.5 |

| Wasserstein | 0.3 |

| CORAL | 0.2 |



\*\*Output files\*\*:

\- `Combined\_Raw\_Shift\_{category}\_{group\_id}.csv` - Fused distance matrix

\- `Heatmap\_{category}\_{group\_id}.png` - Heatmap visualization



\### 3.2 similarity.py - Domain-Level Fusion



\#### `load\_matrix(csv\_path)`

\*\*Purpose\*\*: Load symmetric distance matrix from CSV.



\*\*Args\*\*:

\- `csv\_path` (str): Path to CSV file (first column as index)



\*\*Returns\*\*: `pd.DataFrame` or `None`



\#### `normalize\_matrix(df)`

\*\*Purpose\*\*: Min-Max normalize matrix to \[0, 1] range.



\*\*Args\*\*:

\- `df` (pd.DataFrame): Distance matrix



\*\*Returns\*\*: `pd.DataFrame` - Normalized matrix



\*\*Formula\*\*: `(value - min) / (max - min)`



\#### `distance\_to\_similarity(df, sigma=0.5)`

\*\*Purpose\*\*: Convert distance matrix to similarity using Gaussian RBF kernel.



\*\*Args\*\*:

\- `df` (pd.DataFrame): Normalized distance matrix

\- `sigma` (float): Kernel bandwidth temperature (default: 0.5)



\*\*Returns\*\*: `pd.DataFrame` - Similarity matrix



\*\*Formula\*\*: `sim = exp(-distance^2 / (2 \* sigma^2))`



\*\*Interpretation\*\*:

\- distance = 0 ->similarity = 1.0

\- distance large ->similarity ->0.0



\*\*Output files\*\* (from main execution):

\- `Final\_Similarity\_Matrix.csv` - Fused similarity matrix

\- `Final\_Similarity\_Matrix.png` - Heatmap



\*\*Default weights\*\*: MMD=0.4, Wasserstein=0.4, CORAL=0.2



\### 3.3 similarity\_wsi.py - WSI-Level Fusion



\#### `mmd\_linear(X, Y)`

\*\*Purpose\*\*: Memory-efficient MMD with linear kernel (O(N) complexity).



\*\*Args\*\*:

\- `X` (np.ndarray): Features from slide A, shape (n, d)

\- `Y` (np.ndarray): Features from slide B, shape (m, d)



\*\*Returns\*\*: `float` - MMD² distance



\#### `coral\_dist(X, Y)`

\*\*Purpose\*\*: CORAL distance between two slides.



\*\*Args\*\*:

\- `X` (np.ndarray): Slide A features

\- `Y` (np.ndarray): Slide B features



\*\*Returns\*\*: `float` - Squared Frobenius norm of covariance difference



\#### `wasserstein\_marginal(X, Y)`

\*\*Purpose\*\*: Marginal Wasserstein distance averaged across dimensions.



\*\*Args\*\*:

\- `X` (np.ndarray): Slide A features

\- `Y` (np.ndarray): Slide B features



\*\*Returns\*\*: `float` - Average 1D Wasserstein distance



\#### `normalize\_matrix(mat)`

\*\*Purpose\*\*: Global min-max normalization.



\*\*Args\*\*:

\- `mat` (np.ndarray): Distance matrix



\*\*Returns\*\*: `np.ndarray` - Normalized matrix to \[0, 1]



\#### `dist\_to\_sim(mat, sigma=0.5)`

\*\*Purpose\*\*: Gaussian kernel similarity conversion.



\*\*Args\*\*:

\- `mat` (np.ndarray): Distance matrix

\- `sigma` (float): Kernel bandwidth



\*\*Returns\*\*: `np.ndarray` - Similarity matrix



\#### `load\_data(pkl, csv)`

\*\*Purpose\*\*: Load features and group by slide ID.



\*\*Args\*\*:

\- `pkl` (str): Path to .pkl feature file

\- `csv` (str): Path to midog.csv



\*\*Returns\*\*:

\- `dict`: Keys = slide IDs (as strings), Values = stacked patch features (np.ndarray)



\*\*Constants\*\*:

\- `MAX\_PATCHES\_PER\_SLIDE = 2000` (downsampling for computational efficiency)



\#### `generate\_fused\_matrix(slide\_feats)`

\*\*Purpose\*\*: Compute all three metrics, normalize, convert to similarity, and fuse.



\*\*Args\*\*:

\- `slide\_feats` (dict): From `load\_data`



\*\*Returns\*\*:

\- `tuple`: (final\_similarity\_matrix, sorted\_slide\_ids)



\*\*Process\*\*:

1\. Compute MMD, CORAL, Wasserstein for all slide pairs

2\. Global min-max normalization per metric

3\. Convert distance ->similarity (Gaussian kernel, σ=0.5)

4\. Weighted fusion: MMD=0.5, Wasserstein=0.3, CORAL=0.2

5\. Set diagonal to 1.0 (perfect self-similarity)



\*\*Output files\*\*:

\-`WSI\_Fused\_Similarity\_Matrix.csv` - Fused similarity matrix for all WSIs

\- `WSI\_Fused\_Heatmap.png` - Heatmap with axis ticks hidden for large N



Gemini was used to help format and draft the documentation.md based on my original code. I reviewed and edited all descriptions for technical accuracy.





