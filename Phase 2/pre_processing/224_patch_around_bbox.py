import os
import sys
import json
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

def extract_patches_224(coco_json, image_dir, output_dir, patch_size=224):
    #COCO annotations
    coco = COCO(coco_json)
    img_ids = coco.getImgIds()
    
    #output directory
    os.makedirs(output_dir, exist_ok=True)
    
    patch_info = []
    
    for img_id in img_ids:
        #image info
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(image_dir, img_filename)
        
        #check if image exists
        if not os.path.exists(img_path):
            print(f'Warning: Image {img_path} not found')
            continue
            
        #open image 
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f'Error opening {img_path}: {e}')
            continue
        
        #get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann_idx, ann in enumerate(anns):
            #get bbox [x1, y1, x2, y2]
            bbox = ann['bbox']
            x1, y1, x2, y2 = bbox
            
            #calculate center
            center_x = int((x1+x2)/2)
            center_y = int((y1+y2)/2)
            
            #calculate patch boundaries
            half_size = patch_size//2
            left = max(0, center_x-half_size)
            top = max(0, center_y-half_size)
            right = left+patch_size
            bottom = top+patch_size
            
            #adjust if patch goes beyond image boundaries
            #increase left side if right side is too short
            #vice versa
            if right > img_width:
                right = img_width
                left = right-patch_size
            if bottom > img_height:
                bottom = img_height
                top = bottom-patch_size
            
            patch = img.crop((left, top, right, bottom))
            
            #make sure patch is correct size (in case near edges)
            if patch.size[0] != patch_size or patch.size[1] != patch_size:
                patch = patch.resize((patch_size, patch_size))
            
            patch_name = f"{os.path.splitext(img_filename)[0]}_ann{ann_idx}.tif"
            patch_path = os.path.join(output_dir, patch_name)
            patch.save(patch_path)
            
            #metadata
            patch_info.append({
                'patch_name': patch_name,
                'image_id': img_id,
                'annotation_id': ann['id'],
                'category_id': ann['category_id'],
                'category_name': coco.loadCats(ann['category_id'])[0]['name'],
                'original_bbox': [x1, y1, x2, y2],
                'patch_coords': [left, top, right, bottom]})
            
        print(f'Processed image {img_filename}: {len(anns)} annotations')
    
    #save patch metadata in MScoco file
    with open(os.path.join(output_dir, 'patch_metadata.json'), 'w') as f:
        json.dump(patch_info, f, indent=2)
    
    print(f'\nTotal patches extracted: {len(patch_info)}')
    print(f'Saved to: {output_dir}')
    
    return patch_info

if __name__ == "__main__":
    coco_json = "./images/MIDOGpp.json"
    image_dir = "./images_split/train/"
    output_dir = "./images_split/train/224_patches"
    
    extract_patches_224(coco_json, image_dir, output_dir, patch_size=224)
