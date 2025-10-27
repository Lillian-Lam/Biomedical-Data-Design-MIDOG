# import pandas as pd
# import torch, torchvision
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from torch.utils.data import Dataset
# from ctran import ctranspath


# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# trnsfrms_val = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = mean, std = std)
#     ]
# )
# class roi_dataset(Dataset):
#     def __init__(self, img_csv,
#                  ):
#         super().__init__()
#         self.transform = trnsfrms_val

#         self.images_lst = img_csv

#     def __len__(self):
#         return len(self.images_lst)

#     def __getitem__(self, idx):
#         path = self.images_lst.filename[idx]
#         image = Image.open(path).convert('RGB')
#         image = self.transform(image)


#         return image

# img_csv=pd.read_csv(r'./test_list.csv')
# test_datat=roi_dataset(img_csv)
# database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

# model = ctranspath()
# model.head = nn.Identity()
# td = torch.load(r'./ctranspath.pth')
# model.load_state_dict(td['model'], strict=True)


# model.eval()
# with torch.no_grad():
#     for batch in database_loader:
#         features = model(batch)
#         features = features.cpu().numpy()


import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import pickle
from ctran import ctranspath

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing for CTransPath
# Standard ImageNet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((224, 224)), # CTransPath expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def load_model(checkpoint_path):
    """Load CTransPath model and prepare for feature extraction"""
    model = ctranspath()
    model.head = nn.Identity() # Remove classification head for feature extraction
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    
    model = model.to(device)
    model.eval()
    return model

def extract_features_from_folder(model, image_folder, output_path, batch_size=32):
    """Extract features from all TIFF images in folder"""
    # Get all TIFF files
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob('*.tiff')) + list(image_folder.glob('*.tif'))
    
    print(f"Found {len(image_paths)} TIFF images")
    
    # Extract features
    features_dict = {}
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load and preprocess batch
        for path in batch_paths:
            try:
                # Load image at 40x magnification (0.25 µm/pixel)
                image = Image.open(path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if batch_images:
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Extract features
            with torch.no_grad():
                batch_features = model(batch_tensor)
                batch_features = batch_features.cpu().numpy()
            
            # Store features with filename as key
            for j, path in enumerate(batch_paths[:len(batch_images)]):
                features_dict[path.name] = batch_features[j]
    
    # Save features
    print(f"Saving features to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)
    
    # Also save as numpy array for easier manipulation
    feature_array = np.stack(list(features_dict.values()))
    filenames = list(features_dict.keys())
    
    np.savez_compressed(
        output_path.replace('.pkl', '.npz'),
        features=feature_array,
        filenames=filenames
    )
    
    return features_dict

def main():
    # Configuration
    model_path = './ctranspath.pth'
    image_folder = '/home/caoyang/BDD/MIDOGpp-main/images'
    output_path = './midog_features.pkl'
    
    # Load model
    print("Loading CTransPath model...")
    model = load_model(model_path)
    
    # Extract features
    print("Extracting features from MIDOG++ images...")
    features = extract_features_from_folder(
        model, 
        image_folder, 
        output_path,
        batch_size=32 # Adjust based on GPU memory
    )
    
    # Print summary
    print(f"\nFeature extraction complete!")
    print(f"Total images processed: {len(features)}")
    if features:
        first_key = list(features.keys())[0]
        print(f"Feature dimension: {features[first_key].shape}")
        print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    main()