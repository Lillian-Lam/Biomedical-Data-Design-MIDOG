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
Before running ctranspath_cycleGAN_norm.py, ensure your local environment matches the expected structure:

Image Data: Place your MIDOG++ .tiff files in ./images/.

Metadata: Ensure datasets_xvalidation.csv is located in the ./TransPath/ directory for UMAP generation.

Note: If your data is stored elsewhere (e.g., an external drive), edit the image_folder and model_path variables at the top of the script.


## Setup & Configuration
Before running any extraction script, you must update the path to where your MIDOG++ images are stored on your local machine and download the model weights.  

1. Open the script you wish to run.
2. Update the image_folder variable:

```Python
# CHANGE THIS: Point to your local folder containing .tiff files
image_folder = '/path/to/your/local/MIDOGpp/images'
```
3. This pipeline uses modules from the CTransPath repository and multistain_cyclegan_normalization repository. Clone them into your working directory:
```bash
git clone https://github.com/Xiyue-Wang/TransPath.git
git clone https://github.com/DBO-DKFZ/multistain_cyclegan_normalization.git
```

3. Model Weights:
You must manually download the pretrained weights and place them in the following directories:
- Download ctranspath.pth into ./TransPath/.
- Download latest_net_G_A.pth into ./multistain_cyclegan/resources/weights/.

4. Metadata: Ensure datasets_xvalidation.csv is located in the ./TransPath/ directory for UMAP generation.

5. Update Absolute Paths

```Python
# Use the full absolute path to your data/models
image_folder = '/home/username/data/MIDOGpp/images' 
model_path = '/home/username/project/TransPath/ctranspath.pth'
pretrained_cyclegan_path = '/home/username/project/multistain_cyclegan/resources/weights/latest_net_G_A.pth'
```


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
