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
  
## **Usage**
### **Run full pipeline (default)**
```bash
python ctranspath_cycleGAN_norm.py
```
Runs normalization, feature extraction, and UMAP visualization in sequence.

### **Normalize WSIs only**
```bash
python ctranspath_cycleGAN_norm.py --normalize_images
```
Loads the pretrained CycleGAN model, normalizes all WSIs in `image_folder`, and saves normalized WSIs to `normalized_image_folder`.

### **Extract features only**
```bash
python ctranspath_cycleGAN_norm.py --extract_features
```
If normalization was run previously, uses the normalized WSIs. Also runs UMAP visualization automatically.

### **Skip normalization (use raw WSIs)**
```bash
python ctranspath_cycleGAN_norm.py --skip_normalization
```
Skips the CycleGAN normalization step and runs feature extraction + UMAP on the original WSIs.

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

This pipeline uses modules from the CTransPath repository.
```bash
git clone https://github.com/Xiyue-Wang/TransPath.git
```

This pipeline uses modules from the multistain_cyclegan_normalization repository.
```bash
git clone https://github.com/DBO-DKFZ/multistain_cyclegan_normalization.git
```
