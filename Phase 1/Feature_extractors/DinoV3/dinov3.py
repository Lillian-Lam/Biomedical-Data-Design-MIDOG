import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm.auto import tqdm
import pickle
import warnings

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
# Correct timm model name for DINOv3 Base (Patch 16)
# Reference: https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m
model_name = 'vit_base_patch16_dinov3.lvd1689m' 
if len(sys.argv) < 2:
    print("Usage: python dinov3_feature_extractor.py <path_to_images_directory>")
    sys.exit(1)

image_folder = sys.argv[1]
output_path = './midog_dinov3_features_patches.pkl'

# Patch parameters
patch_size = 224  
stride = 224 # No overlap
max_patches_per_image = 1000  
batch_size = 32  

def get_dinov3_transform(patch_size=224):
    """
    DINOv3 requires standard ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def has_sufficient_tissue(patch, tissue_threshold=0.1):
    """Check if patch contains sufficient tissue"""
    if isinstance(patch, Image.Image):
        patch = np.array(patch)
        
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
    
    if h < patch_size or w < patch_size:
        return [], []

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

def load_model(model_name='vit_base_patch16_dinov3.lvd1689m'):
    """Load DINOv3 model using timm"""
    print(f"Loading {model_name} via timm...")
    
    try:
        import timm
        # Check version implicitly by catching the error if model not found
        # We use num_classes=0 to get the pooled feature vector (CLS token usually)
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        print(f"Successfully loaded {model_name}!")
        
    except ImportError:
        raise RuntimeError("Please install timm: pip install --upgrade timm")
    except RuntimeError as e:
        if "Unknown model" in str(e):
            raise RuntimeError(
                f"Model '{model_name}' not found in timm.\n"
                "CRITICAL: You strictly need timm >= 1.0.20 for DINOv3 support.\n"
                "Please run: pip install --upgrade timm"
            ) from e
        else:
            raise e
        
    model.to(device)
    model.eval()
    
    preprocess = get_dinov3_transform(patch_size)
    return model, preprocess

def extract_features_from_patches(model, preprocess, image_folder, output_path, batch_size=32, 
                                 patch_size=224, stride=224, max_patches_per_image=100):
    """Extract features from patches of all TIFF images in folder"""
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob('*.tiff')) + list(image_folder.glob('*.tif'))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return {}

    print(f"Found {len(image_paths)} TIFF images")
    features_dict = {}
    
    for image_path in tqdm(image_paths, desc="Extracting features"):
        try:
            image = Image.open(image_path).convert('RGB')
            
            patches, coords = extract_tissue_patches(
                image, 
                patch_size=patch_size,
                stride=stride,
                max_patches=max_patches_per_image,
                tissue_threshold=0.1
            )
            
            if not patches:
                continue
            
            patch_features = []
            
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i+batch_size]
                batch_tensors = []
                
                for patch in batch_patches:
                    patch_pil = Image.fromarray(patch)
                    patch_tensor = preprocess(patch_pil)
                    batch_tensors.append(patch_tensor)
                
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors).to(device)
                    
                    with torch.no_grad():
                        # timm with num_classes=0 returns the feature vector directly
                        batch_out = model(batch_tensor)
                        
                        # Normalize features (L2 Norm)
                        batch_out = torch.nn.functional.normalize(batch_out, dim=-1, p=2)
                        
                        batch_features = batch_out.cpu().numpy()
                        patch_features.extend(batch_features)
            
            if len(patch_features) > 0:
                image_features = {
                    'features': np.array(patch_features),
                    'coordinates': coords,
                    'image_size': image.size,
                    'num_patches': len(patches)
                }
                features_dict[image_path.name] = image_features
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Save results
    if features_dict:
        print(f"Saving features to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
            
        # Also save flattened version for easy loading
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
    print(f"Initializing Feature Extraction with DINOv3 ({model_name})...")
    
    model, preprocess = load_model(model_name)
    
    print(f"Model loaded on {device}")
    print("Extracting features from MIDOG++ images...")
    
    features = extract_features_from_patches(
        model, 
        preprocess, 
        image_folder, 
        output_path,
        batch_size=batch_size,
        patch_size=patch_size,
        stride=stride,
        max_patches_per_image=max_patches_per_image
    )
    
    print(f"\nFeature extraction complete!")
    if features:
        first_key = list(features.keys())[0]
        feat_dim = features[first_key]['features'].shape[1]
        print(f"Feature dimension per patch: {feat_dim}")
        print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    main()
