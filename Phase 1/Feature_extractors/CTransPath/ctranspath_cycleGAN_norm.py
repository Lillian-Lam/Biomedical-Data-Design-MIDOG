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
import umap
from sklearn.preprocessing import StandardScaler

#add the TransPath repository from the github (I downloaded it locally)
sys.path.append('./TransPath/')
from ctran import ctranspath

#add MultiStain-CycleGAN repository
sys.path.append('./multistain_cyclegan_normalization')
from models.networks import UnetGenerator, get_norm_layer
from util.util import tensor2im

#configuration from local directory
#CHANGE WHEN YOU RUN YOUR CODE
#model_path = './TransPath/ctranspath.pth'
#pretrained CAMELYON17 CycleGAN model
#downloaded model from from: https://github.com/DBO-DKFZ/multistain_cyclegan_normalization.git (you should clone this repo)
#downloaded the weights from: https://hub.dkfz.de/s/otKYg4onkCNapWT
#placed weights in: ./multistain_cyclegan/resources/weights/latest_net_G_A.pth
#pretrained_cyclegan_path = './multistain_cyclegan/resources/weights/latest_net_G_A.pth'

#image_folder = './images/'
#normalized_image_folder = './images_normalized/'
#output_path = './results_norm/midog_features_patches_normalized.pkl'

#os.makedirs(os.path.dirname(output_path), exist_ok=True)
#os.makedirs(normalized_image_folder, exist_ok=True)

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
   
   
#this is a wrapper for MultiStain-CycleGAN stain normalization using pretrained UnetGenerator, which is based on normalize.py from the MultiStain-CycleGAN repo 
class StainNormalizer:
    def __init__(self, model_weights_path, device='cuda', tile_size=256, tissue_threshold=0):
        self.device = device
        self.tile_size = tile_size #size of the tile to input into generator 
        self.tissue_threshold = tissue_threshold #how much of the tile needs to contain tissue (I just set this to 0)
        
        print(f'Loading pretrained UnetGenerator from: {model_weights_path}')
        
        #create the UnetGenerator model (same as in normalize.py from the repo)
        #UnetGenerator(input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout)
        self.model = UnetGenerator(
            input_nc=3, #RGB input
            output_nc=3, #RGB output
            num_downs=8, #8 downsample/upsample layers
            ngf=64, #number of generator filters
            norm_layer=get_norm_layer('instance'),
            use_dropout=False)
        
        #load pretrained weights from file path
        self.model.load_state_dict(torch.load(model_weights_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        #tranform for CycleGAN (same as in normalize.py from repo)
        #Note:we process at original resolution, but in patches 
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        print(f'Stain normalizer loaded successfully')
    
    #ormalize entire WSI image using CycleGAN. 
    #Uses extract_tissue_patches for tiling with tissue filtering
    def normalize_image(self, image_pil):
        print(f'Normalizing image of size {image_pil.size}')
        
        #extract the tiles
        #there is no limit on max_patches to process all tissue tiles
        tiles, coords = extract_tissue_patches(
            image=image_pil,
            patch_size=self.tile_size,
            stride=self.tile_size,  #no overlap 
            max_patches=10**9,  #process all tissue tiles
            tissue_threshold=self.tissue_threshold)
        
        print(f'Found {len(tiles)} tissue-containing tiles to normalize')
        
        if len(tiles) == 0:
            print('Warning: No tissue found in image, returning original')
            return image_pil
        
        #create output image (start with original to preserve background)
        output_array = np.array(image_pil).copy()
        
        #normalize each tile and place back
        for tile_array, (y_start, x_start, y_end, x_end) in tqdm(
            zip(tiles, coords), 
            total=len(tiles), 
            desc='Normalizing tiles'):
            
            #convert tile to PIL Image
            tile_pil = Image.fromarray(tile_array)
            
            #normalize the tile
            normalized_tile_pil = self._normalize_single(tile_pil)
            normalized_tile_array = np.array(normalized_tile_pil)
            
            #place normalized tile back in output
            output_array[y_start:y_end, x_start:x_end] = normalized_tile_array
        
        return Image.fromarray(output_array)
    
    #normalize a single tile using the UnetGenerator. Based on normalize.py logic
    def _normalize_single(self, image_pil):
        #convert to tensor and normalize to [-1, 1]
        img_tensor = self.normalize_transform(image_pil)
        img_tensor = img_tensor.unsqueeze(0)  
        
        img_tensor = img_tensor.to(self.device)

        #run through UnetGenerator
        with torch.no_grad():
            output = self.model(img_tensor).detach()
        
        #convert back to PIL Image using their utility function
        output_np = tensor2im(output)
        output_pil = Image.fromarray(output_np)
        
        return output_pil

#normalize all the WSI stain images before I move onto feature extraction 
def normalize_all_images(stain_normalizer, input_folder, output_folder):
    #folder paths 
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    #get the number of .tiff WSI images to normalize
    image_paths = list(input_folder.glob('*.tiff'))
    print(f'Found {len(image_paths)} images to normalize')

    
    for image_path in tqdm(image_paths, desc='Normalizing whole images'):
        try:
            #open WSI and covert to RGB
            image = Image.open(image_path).convert('RGB')
            print(f'\nNormalizing {image_path.name}: {image.size}')
            
            #normalize the slide
            normalized_image = stain_normalizer.normalize_image(image)
              
            #output filename (same name, new folder).
            output_path = output_folder / image_path.name
            #saves the normalized WSI as a TIFF with deflate compression.
            normalized_image.save(output_path, compression='tiff_deflate')
            print(f'Saved to {output_path}')
            
        #will continue even if normalization fails for one slide
        except Exception as e:
            print(f'Error normalizing {image_path}: {e}')
            continue

    print(f'\nAll images normalized and saved to {output_folder}')

#from Yang's code
def load_model(checkpoint_path):
    model = ctranspath() #note
    model.head = nn.Identity() # Remove classification head for feature extraction

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    model = model.to(device)
    model.eval()
    return model
    
#Edited Yang's code
def extract_features_from_patches(model, image_folder, output_path, batch_size=32, patch_size=224, stride=224, max_patches_per_image=10000):

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
                print(f"Error processing {image_path}: {e}")
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
def umap_visualizations(features_path):
    # Load features
    with open(features_path, 'rb') as f:
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
        #random_state=42
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

#UMAP analysis
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Input/Output paths
    parser.add_argument('--image_folder', type=str, default='./images/',
                       help='Path to input images folder')
    parser.add_argument('--output_dir', type=str, default='./results_norm',
                       help='Directory to save outputs')
    parser.add_argument('--model_path', type=str, default='./TransPath/ctranspath.pth',
                       help='Path to CTransPath model weights')
    parser.add_argument('--cyclegan_path', type=str, 
                       default='./multistain_cyclegan/resources/weights/latest_net_G_A.pth',
                       help='Path to CycleGAN weights')
    
    #run WSI normalization
    parser.add_argument('--normalize_images', action='store_true', 
                       help='Run stain normalization on images (run this first)')
    #run patch extraction and CTransPath
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract features from (normalized) images')
    #if you want to bypass normalization and just use the raw images
    parser.add_argument('--skip_normalization', action='store_true',
                       help='Skip normalization, use original images')

    # Patch parameters
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Patch size for extraction')
    parser.add_argument('--stride', type=int, default=224,
                       help='Stride for patch extraction')
    parser.add_argument('--max_patches', type=int, default=1000,
                       help='Maximum patches per image')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    args = parser.parse_args()

    #update paths based on arguments
    image_folder = args.image_folder
    normalized_image_folder = f'{args.output_dir}/images_normalized/'
    output_path = f'{args.output_dir}/midog_features_patches_normalized.pkl'
    model_path = args.model_path 
    pretrained_cyclegan_path = args.cyclegan_path  
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(normalized_image_folder, exist_ok=True)

    # Update parameters
    patch_size = args.patch_size
    stride = args.stride
    max_patches_per_image = args.max_patches
    batch_size = args.batch_size
    
    #run everything as default
    if not args.normalize_images and not args.extract_features:
        args.normalize_images = True
        args.extract_features = True

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.normalize_images and not args.skip_normalization:
        print('STEP 1: Normalizing whole images')

        #will give error if they cannot find the normalization model weights
        if not os.path.exists(pretrained_cyclegan_path):
            print(f'\nERROR: Pretrained model not found at: {pretrained_cyclegan_path}')
            sys.exit(1)
            
        #creates the stain normalizer 
        stain_normalizer = StainNormalizer(
        model_weights_path=pretrained_cyclegan_path,
            device=device,
            tile_size=256, #followed the same tile size as normalize.py from the repo
            tissue_threshold=0.0)

        normalize_all_images(stain_normalizer, image_folder, normalized_image_folder)
        print('\nNormalization complete!')

    
    if args.extract_features:
        print('STEP 2: Extracting features from patches')

        if args.skip_normalization:
            feature_image_folder = image_folder
            print('Using original images')
        else:
            feature_image_folder = normalized_image_folder
            print('Using normalized images')

        print('Loading CTransPath model...')
        ctrans_model = load_model(model_path)

        print('\nExtracting features...')
        features = extract_features_from_patches(
            model=ctrans_model,
            image_folder=feature_image_folder,
            output_path=output_path,
            batch_size=batch_size,
            patch_size=patch_size,
            stride=stride,
            max_patches_per_image=max_patches_per_image)

        print('STEP 3: Generating UMAP visualizations')
        
        results_df = umap_visualizations(output_path)
        
        csv_name = 'umap_results_normalized.csv' if not args.skip_normalization else 'umap_results_original.csv'
        results_df.to_csv(csv_name, index=False)
        print(f'\nResults saved to {csv_name}')
