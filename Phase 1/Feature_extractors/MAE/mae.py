import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
import warnings

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Configuration
# Using MAE ViT-Base model (standard architecture for MAE)
model_name = 'mae_vit_base_patch16'
if len(sys.argv) < 2:
    print("Usage: python mae.py <path_to_images_directory>")
    sys.exit(1)

image_folder = sys.argv[1]
output_path = './midog_mae_features_patches.pkl' # Output file for MAE features

# Patch parameters
# MAE standard patch size is 16, but we crop 224x224 regions from the WSI
# The model will internally split this 224 region into 196 tokens (14x14 grid of 16px patches)
patch_size = 224  
stride = 224 # No overlap
max_patches_per_image = 1000  
batch_size = 32  

def get_mae_transform(patch_size=224):
    """
    MAE uses standard ImageNet normalization.
    Input size must be consistent with the model (usually 224x224).
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

def load_model(model_name='mae_vit_base_patch16'):
    model = None
    preprocess = get_mae_transform(patch_size)
    try:
        import timm
        print(f"Attempting to load via timm (using hf-mirror.com)...")
        model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
        model.is_timm = True
        print("Success: Loaded via timm.")
    except Exception as e:
        print(f"Timm load failed: {e}")

    model.to(device)
    model.eval()
    return model, preprocess

def extract_features_from_patches(model, preprocess, image_folder, output_path, batch_size=32,
                                 patch_size=224, stride=224, max_patches_per_image=100):
    """Extract features from patches of all TIFF images"""
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob('*.tiff')) + list(image_folder.glob('*.tif'))
   
    if not image_paths:
        print(f"No images found in {image_folder}")
        return {}

    print(f"Found {len(image_paths)} TIFF images")
    features_dict = {}
   
    # Check if using timm or official repo structure
    is_timm = getattr(model, 'is_timm', False)
   
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
                        if is_timm:
                            # Timm models typically have a forward_features method
                            # Output shape: (B, N, D) -> We take the CLS token at index 0
                            # Or simply use forward_head if available, but manual extraction is safer
                            feats = model.forward_features(batch_tensor)
                            if feats.dim() == 3:
                                # (Batch, Seq_len, Dim) -> Take CLS token (index 0) if exists
                                # MAE usually has a CLS token
                                batch_out = feats[:, 0, :]
                            else:
                                batch_out = feats
                        else:
                            # Official Facebook MAE Repo logic
                            # forward_encoder returns (latent, mask, ids_restore)
                            # We set mask_ratio=0 to encode the full image
                            latent, _, _ = model.forward_encoder(batch_tensor, mask_ratio=0)
                            # latent shape: (Batch, 1 + num_patches, Dim)
                            # The first token (index 0) is the CLS token
                            batch_out = latent[:, 0, :]
                       
                        # L2 Normalization (standard practice for feature comparison)
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
       
        # Save simplified version
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
    print(f"Initializing Feature Extraction with {model_name}...")
   
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
        print(f"Total images processed: {len(features)}")
        first_key = list(features.keys())[0]
        feat_dim = features[first_key]['features'].shape[1]
        print(f"Feature dimension per patch: {feat_dim} (Expected: 768 for ViT-B)")
        print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    main()
