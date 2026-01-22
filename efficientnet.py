import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
model_name = 'efficientnet_b0' # Can be b0, b1, b2, b3, b4
image_folder = '/home/caoyang/BDD/MIDOGpp-main/images'
output_path = './midog_efficientnet_features_patches.pkl'

# Patch parameters
patch_size = 224  
stride = 224 # No overlap
max_patches_per_image = 1000  
batch_size = 32  

# Normalize image with ImageNet mean and std
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def has_sufficient_tissue(patch, tissue_threshold=0.1):
    """Check if patch contains sufficient tissue"""
    # Convert RGB to grayscale if needed
    if len(patch.shape) == 3:
        patch_gray = np.mean(patch, axis=2)
    else:
        patch_gray = patch
    
    # Calculate percentage of non-white pixels
    non_white_pixels = np.sum(patch_gray < 240)
    total_pixels = patch_gray.size
    
    return (non_white_pixels / total_pixels) > tissue_threshold

def extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1):
    """Extract patches containing tissue from image"""
    patches = []
    patch_coords = []
    
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Get image dimensions
    h, w = img_array.shape[:2]
    
    # Number of patches that can fit in each dimension
    h_patches = ((h - patch_size) // stride) + 1
    w_patches = ((w - patch_size) // stride) + 1
    
    # Extract patches
    for i in range(h_patches):
        for j in range(w_patches):
            if len(patches) >= max_patches:
                break
                 
            y_start = i * stride
            x_start = j * stride
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # Extract patch
            patch = img_array[y_start:y_end, x_start:x_end]
        
            # Keep patches that are full size and have sufficient tissue
            if patch.shape[:2] == (patch_size, patch_size) and has_sufficient_tissue(patch, tissue_threshold):
                patches.append(patch)
                patch_coords.append((y_start, x_start, y_end, x_end))
    
    return patches, patch_coords

def load_model(model_name='efficientnet_b0', pretrained=True):
    """Load EfficientNet model and prepare for feature extraction"""
    # Load pretrained EfficientNet
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier = nn.Identity()
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=pretrained)
        model.classifier = nn.Identity()
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained)
        model.classifier = nn.Identity()
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        model.classifier = nn.Identity()
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(pretrained=pretrained)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    model.eval()
    return model

def extract_features_from_patches(model, image_folder, output_path, batch_size=32, 
                                 patch_size=224, stride=224, max_patches_per_image=100):
    """Extract features from patches of all TIFF images in folder"""
    # Get all TIFF files
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob('*.tiff')) + list(image_folder.glob('*.tif'))
    
    print(f"Found {len(image_paths)} TIFF images")
    
    # Extract features
    features_dict = {}
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Extracting features"):
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f'\nProcessing {image_path.name}: {image.size}')
            
            # Extract tissue patches
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
            
            # Process patches in batches
            patch_features = []
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i+batch_size]
                batch_tensors = []
                
                # Convert patches to tensors
                for patch in batch_patches:
                    patch_pil = Image.fromarray(patch)
                    patch_tensor = transform(patch_pil)
                    batch_tensors.append(patch_tensor)
                
                if batch_tensors:
                    # Stack into batch tensor
                    batch_tensor = torch.stack(batch_tensors).to(device)
                    
                    # Extract features
                    with torch.no_grad():
                        batch_features = model(batch_tensor)
                        batch_features = batch_features.cpu().numpy()
                        patch_features.extend(batch_features)
            
            # Store features with metadata
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
    
    # Save features
    print(f"Saving features to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)
    
    # Also save flattened version
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
    # Load model
    print(f"Loading {model_name} model...")
    model = load_model(model_name, pretrained=True)
    
    # Extract features
    print("Extracting features from MIDOG++ images...")
    features = extract_features_from_patches(
        model, 
        image_folder, 
        output_path,
        batch_size=batch_size,
        patch_size=patch_size,
        stride=stride,
        max_patches_per_image=max_patches_per_image
    )
    
    # Print summary
    print(f"\nFeature extraction complete!")
    print(f"Total images processed: {len(features)}")
    if features:
        first_key = list(features.keys())[0]
        print(f"Number of patches for first image: {features[first_key]['num_patches']}")
        print(f"Feature dimension per patch: {features[first_key]['features'].shape[1]}")
        print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    main()
