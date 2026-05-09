import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

np.random.seed(123)

if len(sys.argv) < 2:
    print("Usage: python your_script_name.py <path_to_images_directory>")
    sys.exit(1)

IMAGE_SOURCE_DIR = sys.argv[1]

#get the metadata 
#CHANGE THE DIRECTORY FOR datasets_xvalidation.csv
df = pd.read_csv('../datasets_xvalidation.csv', sep=';').apply(lambda x: x.astype(str).str.strip())
df['Scanner'] = df['Scanner'].replace('Hamammatsu XR', 'Hamamatsu XR')

#get unique slides and create all possible domains
slides = df.drop_duplicates('Slide')
slides['domain'] = slides['Tumor']+'_'+slides['Scanner']+'_'+slides['Origin']+'_'+slides['Species']

#split into train/test/val
train, val, test = [], [], []
for domain in slides['domain'].unique():
    d = slides[slides['domain'] == domain]['Slide'].tolist()
    n_test = max(1, int(np.ceil(len(d) * 0.1)))
    n_val = max(1, int(np.ceil(len(d) * 0.1)))
    test.extend(np.random.choice(d, n_test, replace=False))
    remaining = [s for s in d if s not in test]
    val.extend(np.random.choice(remaining, n_val, replace=False))
    train.extend([s for s in remaining if s not in val])

df[df['Slide'].isin(train)].to_csv('train.csv', index=False, sep=';')
df[df['Slide'].isin(val)].to_csv('val.csv', index=False, sep=';')
df[df['Slide'].isin(test)].to_csv('test.csv', index=False, sep=';')

#create empty folders to put the train and test images
os.makedirs('images_split/train', exist_ok=True)
os.makedirs('images_split/val', exist_ok=True)
os.makedirs('images_split/test', exist_ok=True)


#I wanted to copy the images into a new folder because 
#we might be doing domain shift qunatification etc. in the original folder
def copy_images_to_folders(image_source_dir, train_slides, val_slides, test_slides):
    train_count = 0
    val_count = 0
    test_count = 0

    train_set = set(train_slides)
    val_set = set(val_slides)
    test_set = set(test_slides)
    
    print(f'Looking for images in: {image_source_dir}')
    
    #look through all files in source directory
    for root, dirs, files in os.walk(image_source_dir):
        for file in files:
            #check if it's an .tiff file
            if file.lower().endswith(('.tiff')):
                #try to find slide ID in filename
                file_path = Path(file)
                stem = file_path.stem
            
                #extract numbers from filename
                slide_id = None
                #file starts with numbers (e.g. "001.tiff")
                if stem[0].isdigit():
                    numbers = ''
                    for char in stem:
                        if char.isdigit():
                            numbers += char
                        else:
                            break
                    if numbers:
                        #remove the leading zeros
                        slide_id = str(int(numbers)) 

                #add each WSI to their selected folder
                if slide_id:
                    source_file = os.path.join(root, file)
                    if slide_id in train_set:
                        dest_file = os.path.join('images_split/train', file)
                        shutil.copy2(source_file, dest_file)
                        train_count += 1
                    elif slide_id in val_set:
                        dest_file = os.path.join('images_split/val', file)
                        shutil.copy2(source_file, dest_file)
                        val_count += 1
                    elif slide_id in test_set:
                        dest_file = os.path.join('images_split/test', file)
                        shutil.copy2(source_file, dest_file)
                        test_count += 1
                    #else: slide_id not in our train/val/test sets
    
    return train_count, val_count, test_count 

if os.path.exists(IMAGE_SOURCE_DIR):
    train_img_count, val_img_count, test_img_count = copy_images_to_folders(IMAGE_SOURCE_DIR, train, val, test)
    print(f'Copied {train_img_count} images to images_split/train/')
    print(f'Copied {val_img_count} images to images_split/val/')
    print(f'Copied {test_img_count} images to images_split/test/')
else:
    print(f"Warning: Image source directory '{IMAGE_SOURCE_DIR}' not found.")
