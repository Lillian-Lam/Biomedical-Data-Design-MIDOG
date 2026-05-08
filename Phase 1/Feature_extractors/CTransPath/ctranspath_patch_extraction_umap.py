#packages
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm.auto import tqdm
import pickle
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

#add the TransPath repository from the github (I downloaded it locally)
sys.path.append('./TransPath')
from ctran import ctranspath

#configuration from local directory
#CHANGE WHEN YOU RUN YOUR CODE
model_path = './TransPath/ctranspath.pth'
image_folder = './images'
output_path = './midog_features_patches.pkl'

#patch parameters (we want to extract features in patches rather than resize the image)
patch_size = 224
stride = 224      #it moves by 224 pixels, so no overlap
max_patches_per_image = 1000  #limit patches (I started with 100 to test if my code works)
batch_size = 32

#normalize image with mean and std of Imagenet
#cTransPath is trained on normalized images (states to do it here https://huggingface.co/kaczmarj/CTransPath)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

def has_sufficient_tissue(patch, tissue_threshold=0.1):
    #convert RGB to grayscale if needed
    if len(patch.shape) == 3:
        #take mean across color channels to get grayscale
        patch_gray = np.mean(patch, axis=2)
    else:
        patch_gray = patch

    #calculate percentage of non-white pixels
    #background/empty areas are bright/white (no stain)
    non_white_pixels = np.sum(patch_gray < 240) #pixels darker than white threshold
    total_pixels = patch_gray.size

    #we want non-white pixels exceed threshold
    return (non_white_pixels / total_pixels) > tissue_threshold

def extract_tissue_patches(image, patch_size=224, stride=224, max_patches=100, tissue_threshold=0.1):
    patches = []
    patch_coords = [] #coordinates of each patch

    #convert PIL Image to numpy array for array operations
    #we load the images in from a .tiff to a PIL image object
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    #get image dimensions (height, width)
    h, w = img_array.shape[:2]

    #number of patches that can fit in each dimension
    h_patches = ((h-patch_size)//stride)+1
    w_patches = ((w-patch_size)//stride)+1


    #extract patches
    for i in range(h_patches): #top to bottom
        for j in range(w_patches): #left to right
            if len(patches) >= max_patches: #limit number of patches
                break

            y_start = i*stride #top edge of patch
            x_start = j*stride #left edge of patch
            y_end = y_start+patch_size #bottom edge
            x_end = x_start+patch_size #right edge

            #array slicing to get patch
            patch = img_array[y_start:y_end, x_start:x_end]

            #keep patches that are full size and have sufficient tissue
            if patch.shape[:2] == (patch_size, patch_size) and has_sufficient_tissue(patch, tissue_threshold):
                patches.append(patch)
                patch_coords.append((y_start, x_start, y_end, x_end))

    return patches, patch_coords

#from Yang's code
def load_model(checkpoint_path):
    """Load CTransPath model and prepare for feature extraction"""
    model = ctranspath() #note
    model.head = nn.Identity() # Remove classification head for feature extraction

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)

    model = model.to(device)
    model.eval()
    return model

#Edited Yang's code
def extract_features_from_patches(model, image_folder, output_path, batch_size=32, patch_size=224, stride=224, max_patches_per_image=100):
    """Extract features in patches from all TIFF images in folder"""
    # Get all TIFF files
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob('*.tiff'))  # finds files ending with .tiff (note: they are all .tiff files)

    print(f"Found {len(image_paths)} TIFF images")

    #extract features in dictionary (structure: {filename: feature_vector})
    features_dict = {}

    # Process in batches for efficiency
    for image_path in tqdm(image_paths, desc="Extracting features"):
        try:
            image = Image.open(image_path).convert('RGB') #open image and convert to RGB
            print(f'\nProcessing {image_path.name}: {image.size}')

            #get the image patches with tissue filtering
            patches, coords = extract_tissue_patches(image,
                                                     patch_size=patch_size,
                                                     stride=stride,
                                                     max_patches=max_patches_per_image,
                                                     tissue_threshold=0.1)
            print(f'Extracted {len(patches)} tissue patches')

            if not patches: #skip image if no tissue patches were found
                continue

            #process patches in batches rather than just the whole image
            patch_features = []
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i+32] #slices the patch list to get current batch
                batch_tensors = []

                # convert patch in the batch to tensor
                for patch in batch_patches:
                    patch_pil = Image.fromarray(patch) #convert numpy array back to PIL Image
                    patch_tensor = transform(patch_pil) #just normalize the patch
                    batch_tensors.append(patch_tensor)

                #Follows Yang's code
                #we only process if some patches have successfully converted
                if batch_tensors:
                    #stack each patch tensors into a single batch tensor
                    batch_tensor = torch.stack(batch_tensors).to(device)

                    #extracting the features
                    with torch.no_grad():
                        batch_features = model(batch_tensor) #extract features with gradients disabled
                        batch_features = batch_features.cpu().numpy() #convert to numpy array
                        #we add these batch features to our collection for this image
                        patch_features.extend(batch_features)

            #store features with patch coordinates
            image_features = {'features': np.array(patch_features), #feature vector
                              'coordinates': coords, #patch location
                              'image_size': image.size, #original WSI size
                              'num_patches': len(patches)} #number of patches extracted

            # Store features with filename as key
            features_dict[image_path.name] = image_features

        except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

    #Yang's code
    # Save features (to a pickle file)
    print(f"Saving features to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)

    # Also save as numpy array for easier manipulation
    #combines all patches from all images into one big array
    all_features = []
    all_filenames = []
    all_patch_ids = []

    for filename, data in features_dict.items():
        for patch_idx in range(data['features'].shape[0]): #iterate for each patch
            all_features.append(data['features'][patch_idx]) #feature vector for the patch
            all_filenames.append(filename) #which image it came from
            all_patch_ids.append(patch_idx) #which patch number in that image

    #save if we have features
    if all_features:
        feature_array = np.array(all_features) #convert features list to 2D numpy array

        #save the compressed numpy file
        np.savez_compressed(
            output_path.replace('.pkl', '_flat.npz'),
            features=feature_array,
            filenames=all_filenames,
            patch_ids=all_patch_ids)

    return features_dict

#edited from Yang's code
def umap_visualizations():
    # Load features
    with open('./midog_features_patches.pkl', 'rb') as f:
        features_dict = pickle.load(f)

    #load metadata from csv file
    metadata_df = pd.read_csv('TransPath/datasets_xvalidation.csv', sep=';')
    metadata_df.columns = metadata_df.columns.str.strip()
    metadata_df['Slide'] = metadata_df['Slide'].astype(str).str.strip()
    train_metadata_df = metadata_df[metadata_df['Dataset'] == 'train'].copy() #get only the training

    all_features = []
    all_filenames = []
    slide_numbers = []

    #get the slide numbers from filenames to merge with metadata
    # Extract slide numbers from filenames (e.g., '034.tiff' -> '34')
    for filename, data in features_dict.items():
        base_name = Path(filename).stem #remove the file extension
        try:
            slide_num = str(int(base_name)) #convert to int then string to remove leading zeros
        except ValueError:
            slide_num = base_name #keep original if it does not work

        # Only include features from training slides
        if slide_num in train_metadata_df['Slide'].values:
            features = data['features'] #feature vectors for image
            all_features.extend(features)
            all_filenames.extend([filename]*len(features)) #repeat filename for each patch
            slide_numbers.extend([slide_num]*len(features))

    feature_array = np.array(all_features)

    # UMAP embedding
    #standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_array)

    #I just followed the same parameter as Yang
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2, #we reduce to 2 dimensions
        min_dist=0.1, #how close clusters are
        metric='cosine', #distance metric (I think cosine is better for high dimensional data)
        random_state=42
    )
    embedding = umap_model.fit_transform(features_scaled)

    #DataFrame with embeddings and metadata
    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    embedding_df['filename'] = all_filenames
    embedding_df['Slide'] = slide_numbers

    #merge with metadata
    final_df = embedding_df.merge(metadata_df[['Slide', 'Dataset', 'Tumor', 'Scanner', 'Origin', 'Species']],
                                  on='Slide',
                                  how='inner') #inner join to ensure only training data remains

    #visualizations for each category
    categories = ['Slide', 'Tumor', 'Scanner', 'Origin', 'Species']

    #following Yang's code for plotting
    for category in categories:
        plt.figure(figsize=(14, 10)) #new figure for each category

        unique_values = sorted(final_df[category].dropna().unique()) #get unique values for this category
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values))) # getdistinct colors for each value

        for value, color in zip(unique_values, colors): #plot each group with a different color
            mask = final_df[category] == value #boolean mask for this to select patches with specific value
            plt.scatter(
                final_df.loc[mask, 'UMAP1'],
                final_df.loc[mask, 'UMAP2'],
                label=str(value),
                color=color,
                s=60,
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5
            )

        plt.title(f'UMAP Projection of Features, Colored by {category}', fontsize=16, pad=20)
        plt.xlabel('UMAP Component 1', fontsize=14)
        plt.ylabel('UMAP Component 2', fontsize=14)
        if category != 'Slide':
            plt.legend(title=category, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'umap_{category.lower()}_patches.png', dpi=300, bbox_inches='tight')
    return final_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #run patch extraction and CTransPath
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract features from images')
    #run UMAP visualization
    parser.add_argument('--umap', action='store_true',
                       help='Generate UMAP visualizations from extracted features')
    args = parser.parse_args()

    #run everything as default
    if not args.extract_features and not args.umap:
        args.extract_features = True
        args.umap = True

    #I am running this locally on my computer (cpu), but we would want gpu processing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.extract_features:
        print('STEP 1: Extracting features from patches')

        print('Loading CTransPath model...')
        model = load_model(model_path)

        print('\nExtracting features...')
        features = extract_features_from_patches(model,
                                                 image_folder,
                                                 output_path,
                                                 batch_size=batch_size,
                                                 patch_size=patch_size,
                                                 stride=stride,
                                                 max_patches_per_image=max_patches_per_image)

    if args.umap:
        print('STEP 2: Generating UMAP visualizations')

        #UMAP analysis
        results_df = umap_visualizations()
        results_df.to_csv('umap_results_with_metadata.csv', index=False)
        print('\nResults saved to umap_results_with_metadata.csv')
