# **WSI Stain Normalization + CTransPath Feature Extraction**
A complete pipeline for:
- Whole‑slide stain normalization using a CycleGAN model  
- Patch extraction from normalized WSIs  
- Feature extraction using CTransPath 
- UMAP visualization of patch‑level embeddings

## **Pipeline Overview**
### **1. Whole‑slide stain normalization (CycleGAN)**
- The WSI is tiled into 256×256 non‑overlapping tiles  
- Each tile is normalized using a pretrained CycleGAN generator  
- Tiles are reassembled into a full normalized WSI  
- Output is saved as a compressed TIFF  

### **2. Patch extraction (CTransPath‑ready)**
- Extract 224×224 patches from the normalized WSI    
- Tissue detection filters out background  

### **3. Feature extraction (CTransPath)**
- Each patch is transformed with ImageNet mean/std  
- Passed through CTransPath to obtain a feature vector  
- Features + coordinates are saved as pickle file

### **4. UMAP visualization**
- Runs automatically after feature extraction  
- Computes 2D UMAP embedding  
- Saves per‑category scatter plots and results as a CSV

## Setup & Configuration
Before running any extraction script, you must update the path to where your MIDOG++ images are stored on your local machine and download the model weights.  

1. This pipeline uses modules from the CTransPath repository and multistain_cyclegan_normalization repository. Clone them into your working directory:
```bash
git clone https://github.com/Xiyue-Wang/TransPath.git
git clone https://github.com/DBO-DKFZ/multistain_cyclegan_normalization.git
```

2. Model Weights:
You must manually download the pretrained weights and place them in the following directories:
- Download ctranspath.pth into ./TransPath/.
- Download latest_net_G_A.pth into ./multistain_cyclegan/resources/weights/.

3. Metadata: Ensure datasets_xvalidation.csv is located in the ./TransPath/ directory for UMAP generation.

## **Usage**
### **Run full pipeline (default)**
```bash
python ctranspath_cycleGAN_norm.py \
  --image_folder /path/to/your/images \
  --output_dir /path/to/results \
  --model_path /path/to/ctranspath.pth \
  --cyclegan_path /path/to/latest_net_G_A.pth
```
Runs normalization, feature extraction, and UMAP visualization in sequence.

### **Normalize WSIs only**
```bash
python ctranspath_cycleGAN_norm.py \
  --normalize_images \
  --image_folder /path/to/your/images \
  --output_dir /path/to/results \
  --model_path /path/to/ctranspath.pth \
  --cyclegan_path /path/to/latest_net_G_A.pth
```
Loads the pretrained CycleGAN model, normalizes all WSIs in image_folder, and saves normalized WSIs to `{output_dir}/images_normalized/`.

### **Extract features only**
```bash
python ctranspath_cycleGAN_norm.py \
  --extract_features \
  --output_dir /path/to/results
```
If normalization was run previously, uses the normalized WSIs from `{output_dir}/images_normalized/`. Also runs UMAP visualization automatically

### **Skip normalization (use raw WSIs)**
```bash
python ctranspath_cycleGAN_norm.py \
  --skip_normalization \
  --extract_features \
  --image_folder /path/to/your/images \
  --output_dir /path/to/results
```
Skips the CycleGAN normalization step and runs feature extraction + UMAP on the original WSIs.

### **Custom patch parameters**
```bash
python ctranspath_cycleGAN_norm.py \
  --patch_size 224 \
  --stride 224 \
  --max_patches 500 \
  --batch_size 64
```

## **Output**
### **Normalized WSIs**
```
normalized_image_folder/
    slide1.tiff
    slide2.tiff
    ...
```

### **Feature Files**
```
results_norm/
    midog_features_patches_normalized.pkl
    midog_features_patches_normalized_flat.npz
```

### **UMAP Results**
```
umap_slide_patches.png
umap_tumor_patches.png
umap_scanner_patches.png
umap_origin_patches.png
umap_species_patches.png
umap_results_normalized.csv      # default
umap_results_original.csv        # when --skip_normalization is used
```

## **Environment Setup**
This project requires Python 3.9+ and a working PyTorch installation (GPU recommended). This project is not compatible with Python 3.14+ as some dependencies have not been updated for the latest Python releases.
