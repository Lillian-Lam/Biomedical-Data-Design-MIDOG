
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
- Loads extracted features  
- Computes 2D UMAP embedding  
- Saves results as a CSV
  
## **Usage**

### **Normalize WSIs**
```bash
python ctranspath_cycleGAN_norm.py --normalize_images
```
This will load the pretrained CycleGAN model, normalize all WSIs in `image_folder`, and save normalized WSIs to `normalized_image_folder`  

### **Extract features**
```bash
python ctranspath_cycleGAN_norm.py --extract_features
```
If normalization was run previously, this uses the normalized WSIs.

### **Skip normalization (use raw WSIs)**
```bash
python ctranspath_cycleGAN_norm.py --extract_features --skip_normalization
```

### **Run full pipeline (default)**
```bash
python ctranspath_cycleGAN_norm.py
```
### **Normalized WSIs**
```
normalized_image_folder/
    slide1.tiff
    slide2.tiff
    ...
```
## **Output**
### **Feature Files**
```
results_norm/
    midog_features_patches_normalized.pkl
    midog_features_patches_normalized_flat.npz
```

### **UMAP Results**
```
umap_results_normalized.csv
```

## Environment Setup

This project requires Python 3.9+ and a working PyTorch installation (GPU recommended). This project is not compatible with Python 3.14+ as some dependencies have not been updated for the latest Python releases.

This pipeline uses modules from the CTransPath repository.
``` bash
git clone https://github.com/Xiyue-Wang/TransPath.git
```

This pipeline uses modules from the multistain_cyclegan_normalization repository.
``` bash
git clone https://github.com/DBO-DKFZ/multistain_cyclegan_normalization.git
```
