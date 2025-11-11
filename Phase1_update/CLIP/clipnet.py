import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms # We still need this for ToTensor
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import clip

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
model_name = 'ViT-B/32' # CLIP model
image_folder = '/home/caoyang/BDD/MIDOGpp-main/images'
output_path = './midog_clip_features_patches.pkl'

# Patch parameters
patch_size = 224  
stride = 224 # No overlap
max_patches_per_image = 1000  
batch_size = 32  

# CLIP has its own transform, which we load with the model.
# We will load it in load_model and pass it to the extraction function.
# We delete the global 'transform' variable.

def has_sufficient_tissue(patch, tissue_threshold=0.1):
    """Check if patch contains sufficient tissue"""
    if len(patch.shape) == 3:
        patch_gray = np.mean(patch, axis=2)
    else:
        patch_gray = patch
    non_white_pixels = np.sum(patch_gray < 240)
    total_pixels = patch_gray.size
    return (non_white_pixels / total_pixels) > tissue_threshold

def extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1):
    """Extract patches containing tissue from image"""
    patches = []
    patch_coords = []
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    h, w = img_array.shape[:2]
    h_patches = ((h - patch_size) // stride) + 1
    w_patches = ((w - patch_size) // stride) + 1
    
    for i in range(h_patches):
        for j in range(w_patches):
            if len(patches) >= max_patches:
                break
            y_start = i * stride
            x_start = j * stride
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            patch = img_array[y_start:y_end, x_start:x_end]
            if patch.shape[:2] == (patch_size, patch_size) and has_sufficient_tissue(patch, tissue_threshold):
                patches.append(patch)
                patch_coords.append((y_start, x_start, y_end, x_end))
    return patches, patch_coords

def load_model(model_name='ViT-B/32'):
    """Load CLIP model and its preprocessor"""
    model, preprocess = clip.load(model_name, device=device, jit=True) 
    model.eval()
    return model, preprocess

def extract_features_from_patches(model, preprocess, image_folder, output_path, batch_size=32, 
                                 patch_size=224, stride=224, max_patches_per_image=100):
    """Extract features from patches of all TIFF images in folder"""
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob('*.tiff')) + list(image_folder.glob('*.tif'))
    print(f"Found {len(image_paths)} TIFF images")
    features_dict = {}
    
    for image_path in tqdm(image_paths, desc="Extracting features"):
        try:
            image = Image.open(image_path).convert('RGB')
            print(f'\nProcessing {image_path.name}: {image.size}')
            
            patches, coords = extract_tissue_patches(
                image, 
                patch_size=patch_size,
                stride=stride,
                max_patches=max_patches_per_image,
                tissue_threshold=0.1
            )
            print(f'Extracted {len(patches)} tissue patches')
            
            if not patches:
                continue
            
            patch_features = []
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i+batch_size]
                batch_tensors = []
                
                for patch in batch_patches:
                    patch_pil = Image.fromarray(patch)
                    
                    # Apply CLIP's specific preprocess function
                    patch_tensor = preprocess(patch_pil)
                    
                    batch_tensors.append(patch_tensor)
                
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors).to(device)
                    
                    with torch.no_grad():
                        # Use model.encode_image to get features
                        batch_features = model.encode_image(batch_tensor)
                        # Normalize features (good practice for CLIP)
                        batch_features /= batch_features.norm(dim=-1, keepdim=True)
                        batch_features = batch_features.float().cpu().numpy() 
                        patch_features.extend(batch_features)
            
            image_features = {
                'features': np.array(patch_features),
                'coordinates': coords,
                'image_size': image.size,
                'num_patches': len(patches)
            }
            features_dict[image_path.name] = image_features
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"Saving features to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)
    
    all_features = []
    all_filenames = []
    all_patch_ids = []
    
    for filename, data in features_dict.items():
        for patch_idx in range(data['features'].shape[0]):
            all_features.append(data['features'][patch_idx])
            all_filenames.append(filename)
            all_patch_ids.append(patch_idx)
    
    if all_features:
        feature_array = np.array(all_features)
        np.savez_compressed(
            output_path.replace('.pkl', '_flat.npz'),
            features=feature_array,
            filenames=all_filenames,
            patch_ids=all_patch_ids
        )
    return features_dict

def main():
    print(f"Loading {model_name} model...")
    model, preprocess = load_model(model_name)
    
    print("Extracting features from MIDOG++ images...")
    features = extract_features_from_patches(
        model, 
        preprocess, # Pass preprocess function
        image_folder, 
        output_path,
        batch_size=batch_size,
        patch_size=patch_size,
        stride=stride,
        max_patches_per_image=max_patches_per_image
    )
    
    print(f"\nFeature extraction complete!")
    print(f"Total images processed: {len(features)}")
    if features:
        first_key = list(features.keys())[0]
        print(f"Number of patches for first image: {features[first_key]['num_patches']}")
        print(f"Feature dimension per patch: {features[first_key]['features'].shape[1]}")
        print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    main()