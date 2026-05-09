"""
cell_segmentation_to_coco.py

This script processes whole slide images (WSI) with Cellpose segmentation.
Outputs bounding boxes for each cell in MS COCO format.
"""
# pip install cellpose 
# pip install scikit-image

import os
import json
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, simpledialog
import gc

# Handle missing cellpose
try:
    from cellpose import models
except ImportError:
    raise ImportError("cellpose is not installed. Please run: pip install cellpose")

# Handle missing scikit-image
try:
    from skimage.measure import regionprops, label
except ImportError:
    raise ImportError("scikit-image is not installed. Please run: pip install scikit-image")

# Directory (adjust as needed)
# Processing full whole slide images (7.2K x 5.4K)
WSI_DIR = "./images"  # Full WSI images
OUTPUT_DIR = "./Phase 2/preprocessing/cellpose_annotations" # Output directory

# Helper to get all WSI files
def get_wsi_files(wsi_dir):
    """Get all TIFF WSI files from train directory."""
    exts = ('.tif', '.tiff')
    files = []
    for f in sorted(os.listdir(wsi_dir)):
        if f.lower().endswith(exts):
            full_path = os.path.join(wsi_dir, f)
            if os.path.isfile(full_path):
                files.append(full_path)
    return files

# Segmentation
# Global model instance to avoid reloading
_cellpose_model = None

def get_cellpose_model():
    """Get or initialize the Cellpose model."""
    global _cellpose_model
    if _cellpose_model is None:
        _cellpose_model = models.CellposeModel(model_type='cyto')
    return _cellpose_model

def segment_cells(img_np):
    """Run Cellpose segmentation on a single patch."""
    try:
        model = get_cellpose_model()
        # CellposeModel.eval() returns (masks, flows, styles)
        masks, _, _ = model.eval(img_np, diameter=None, channels=[[2,1,0]])
        return masks
    except Exception as e:
        print(f"  Warning: Segmentation failed - {e}")
        return None

# Bounding Boxes
def masks_to_bboxes(masks):
    """Convert segmentation masks to bounding boxes."""
    if masks is None or masks.max() == 0:
        return []
    
    bboxes = []
    labeled = label(masks)
    for region in regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        bboxes.append([int(minc), int(minr), int(maxc), int(maxr)])
    return bboxes

# Coco Output
def save_coco(image_name, bboxes, img_width, img_height, image_id=1):
    """Save annotations in MS COCO format for WSI."""
    annotations = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        annotations.append({
            "id": i+1,
            "image_id": image_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "category_id": 1,
            "iscrowd": 0
        })
    
    coco = {
        "images": [{
            "id": image_id,
            "file_name": image_name,
            "width": img_width,
            "height": img_height
        }],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cell"}]
    }
    return coco

# Main Function Defined

def main():
    wsi_dir = WSI_DIR
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    wsi_files = get_wsi_files(wsi_dir)
    print(f"Found {len(wsi_files)} WSI images in {wsi_dir}")
    
    successful = 0
    failed = 0
    empty = 0
    
    for idx, wsi_path in enumerate(wsi_files):
        wsi_name = os.path.basename(wsi_path)
        print(f"\n[{idx+1}/{len(wsi_files)}] Processing {wsi_name}...", flush=True)
        
        try:
            # Load WSI
            img = Image.open(wsi_path)
            print(f"  Image size: {img.size}, mode: {img.mode}")
            
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            img_np = np.array(img)
            print(f"  Numpy shape: {img_np.shape}, dtype: {img_np.dtype}")
            
            # Handle grayscale - convert to RGB format that is expected by Cellpose
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np, img_np, img_np], axis=2)
            
            # Run segmentation
            print(f"  Running Cellpose segmentation...", flush=True)
            masks = segment_cells(img_np)
            if masks is None:
                failed += 1
                print("  segmentation failed")
                continue
            
            # Get bounding boxes
            bboxes = masks_to_bboxes(masks)
            print(f"  Found {len(bboxes)} cells")
            
            if len(bboxes) == 0:
                empty += 1
                print("  no cells detected")
                continue
            
            # Save COCO format
            coco_data = save_coco(wsi_name, bboxes, img.width, img.height, image_id=idx+1)
            out_path = os.path.join(output_dir, os.path.splitext(wsi_name)[0] + "_wsi_coco.json")
            
            with open(out_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            successful += 1
            print(f"  ✓ Saved to {os.path.basename(out_path)}")
            
            # Free memory
            del img, img_np, masks, coco_data
            gc.collect()
            
        except Exception as e:
            failed += 1
            print(f"  error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Empty (no cells): {empty}")
    print(f"  Failed: {failed}")
    print(f"  Total processed: {len(wsi_files)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
