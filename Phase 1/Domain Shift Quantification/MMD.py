#packages required
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

def mmd_unbiased(X, Y, use_scaled=True):
    #get the number of samples in each domain
    n, m = len(X), len(Y)

    if use_scaled:
        #scale with pooled statistics
        pooled = np.vstack([X, Y])
        #apply the pooled std if it is positive
        #otherwise the data is not scaled 
        scale_factor = np.std(pooled) if np.std(pooled) > 0 else 1.0
        X = X / scale_factor
        Y = Y / scale_factor

    #linear kernel (dot product)
    #similarity within domain
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    #similarity between domain
    XY = np.dot(X, Y.T)

    #exclude diagonal elements
    #follows formula for unbiased estimate of MMD^2
    mmd_sq = (XX.sum() - np.trace(XX))/(n * (n-1)) + \
             (YY.sum() - np.trace(YY))/(m * (m-1)) - \
             2*XY.mean()

    return mmd_sq

def mmd_intra_domain(X, n_splits=10, use_scaled=True):
    #we calculate the intra-domain MMD by splitting data and comparing halves

    #if the number of samples is small, the splits would also be small
    #I don't think it would be practical to do MMD
    if len(X) < 10:
        return np.nan  

    mmds = []
    #want to split the data into 2 halves multiple times
    for _ in range(n_splits):
        #random permutation to shuffle the data
        indices = np.random.permutation(len(X))

        #split into halves
        mid = len(X) // 2
        X1 = X[indices[:mid]]
        X2 = X[indices[mid:2*mid]]  #ensure equal sizes

        #compute the mmd for each half
        mmd = mmd_unbiased(X1, X2, use_scaled=use_scaled)
        mmds.append(mmd)

    #averages MMD across n_splits random splits
    return np.mean(mmds)


def load_and_prepare_data(features_path, metadata_path):
    #we load the CTransPath extracted features from the pickle file 
    with open(features_path, 'rb') as f:
        features_dict = pickle.load(f)

    #metadata from the MIDOG++ website 
    #contains the data for each WSI .tiff 
    metadata_df = pd.read_csv(metadata_path, sep=';')
    #data cleaning
    #remove whitespace in column names
    metadata_df.columns = metadata_df.columns.str.strip()
    #standardize the slides names as strings without whitespace
    metadata_df['Slide'] = metadata_df['Slide'].astype(str).str.strip()
    #filter only the training dataset out
    train_df = metadata_df[metadata_df['Dataset'] == 'train'].copy()
    #change the typo within the dataset
    train_df = train_df.replace('Hamammatsu XR', 'Hamamatsu XR')

    #list for all of the data
    all_features = [] #feature vectors at patch level
    all_filenames = [] #source file for each patch 
    slide_numbers = [] #slide identifier for each patch
    scanners = [] #the scanner model 
    tumor_types = [] 
    origins = [] #lab origin
    species_list = [] #human or canine

    #process each slide file
    for filename, data in features_dict.items():
        #filename without the extension
        base_name = Path(filename).stem
        slide_num = str(int(base_name))  #convert "001" to "1"

        #check if slide exists in metadata
        if slide_num in train_df['Slide'].values:
            #get metadata for the slide
            slide_meta = train_df[train_df['Slide'] == slide_num].iloc[0]
            #get feature vectors for all patches in the slide
            features = data['features']

            #
            if len(features) > 0:
                #add all patch features
                all_features.extend(features)
                #repeat filename for each patch
                all_filenames.extend([filename] * len(features))
                #repeat slide ID for each patch
                slide_numbers.extend([slide_num] * len(features))
                #repeat scanner name for each patch
                scanners.extend([slide_meta['Scanner']] * len(features))
                #repeat tumor type for each patch
                tumor_types.extend([slide_meta['Tumor']] * len(features))
                #repeat lab origin for each patch
                origins.extend([slide_meta['Origin']] * len(features))
                ##repeat species category for each patch
                species_list.extend([slide_meta['Species']] * len(features))

    #covert all features into numpy array
    feature_array = np.array(all_features)
    #create the dataframe with the metadata columns
    features_df = pd.DataFrame({
        'filename': all_filenames,
        'slide': slide_numbers,
        'scanner': scanners,
        'tumor_type': tumor_types,
        'origin': origins,
        'species': species_list})
    #store the features vectors as a seperate column 
    features_df['features'] = list(feature_array)
    #combined dataset with the feature vectors and metadata
    return features_df


def analyze_mmd_differences(features_df, category_column, category_name, save_plot=True, output_dir='mmd_results',  data_type='raw'):
    #analyze distribution differences betwwen categories with mmd
    #get the unique category for comparision 
    categories = features_df[category_column].unique()

    #extract the features for each category group in dictionary format
    category_features = {}
    for category in categories:
        #filter for only rows belonging in the desired category
        category_data = features_df[features_df[category_column] == category]
        #process only if category has data
        if len(category_data) > 0:
            #convert the feature vectors into numpy array 
            features = np.array(category_data['features'].tolist())
            category_features[category] = features

    #create MMD matrix to be filled in 
    n_categories = len(categories)
    mmd_matrix = np.zeros((n_categories, n_categories))

    #calculate the MMD distances
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            #the diagonals 
            if i == j:
                #Intra-domain MMD 
                X = category_features[cat1]
                mmd_matrix[i, j] = mmd_intra_domain(X, n_splits=10, use_scaled=True)
            #only compute upper triangle (optimize as mmd should be the same for lower triangle)
            elif i < j:  
                #cross-domain mmd between different categories
                X = category_features[cat1]
                Y = category_features[cat2]
                mmd_matrix[i, j] = mmd_unbiased(X, Y, use_scaled=True)
                #mirror to lower triangle (symmetric matrix)
                mmd_matrix[j, i] = mmd_matrix[i, j]

    #heatmap of MMD distances
    fig, ax = plt.subplots(figsize=(10, 8))
    mmd_df = pd.DataFrame(mmd_matrix, index=categories, columns=categories)
    sns.heatmap(mmd_df, annot=True, cmap='YlOrRd', ax=ax, fmt='.4f',
                cbar_kws={'label': 'MMD Distance'})
    ax.set_title(f'MMD Distance Between {category_name}s ({data_type} data)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_plot:
        #create the directory 
        os.makedirs(output_dir, exist_ok=True)  
        filename = f"{output_dir}/mmd_heatmap_{category_column}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'Heatmap saved to: {filename}')
        plt.savefig(f'mmd_heatmap_{category_column}.png', dpi=300, bbox_inches='tight')

    plt.show()

    return mmd_matrix, categories


feature_files = {
    'raw': './midog_feature_patches.pkl',
    'normalized': './midog_features_patches_normalized.pkl'}

metadata_path = './datasets_xvalidation.csv'


if __name__ == '__main__':
    for data_type, feature_path in feature_files.items():
        #load dataset
        df = load_and_prepare_data(feature_path, metadata_path)

        #running the different control tests 
        tests = [
            {'name': f'Tumor Type (Hamamatsu XR only) [{data_type}]',
             'data': df[df['scanner'] == 'Hamamatsu XR'],
             'cat_col': 'tumor_type',
             'cat_name': 'Tumor Type', 
             'data_type': data_type},
            {'name': f'Scanner (Human Breast Cancer only) [{data_type}]',
             'data': df[df['tumor_type'] == 'human breast cancer'],
             'cat_col': 'scanner',
             'cat_name': 'Scanner', 
             'data_type': data_type},
            {'name': f'Origin (Canine Soft Tissue Sarcoma only) [{data_type}]',
             'data': df[df['tumor_type'] == 'canine soft tissue sarcoma'],
             'cat_col': 'origin',
             'cat_name': 'Origin',
             'data_type': data_type}]
        for test in tests:
          print(f'{test['name']}')
    
          mmd_matrix, cats = analyze_mmd_differences(
          test['data'], 
          test['cat_col'], 
          test['cat_name'],
          save_plot=True,
          output_dir='mmd_results',
          data_type=test['data_type'])


