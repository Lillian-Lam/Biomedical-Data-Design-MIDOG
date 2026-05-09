# Phase 2 - Function Documentation

## Overview

This document provides detailed function-level documentation for all Phase 2 scripts, including input/output expectations, argument types, and return values.

## 1. cell_segmentation_to_coco.py

### `get_wsi_files(wsi_dir)`

**Purpose**: Get all TIFF WSI files from a directory.

**Args**:
- `wsi_dir` (str): Path to directory containing WSI files

**Returns**:
- `list` of str: Full paths to all `.tif` and `.tiff` files in the directory

### `get_cellpose_model()`

**Purpose**: Get or initialize the global Cellpose model instance (singleton pattern).

**Args**: None

**Returns**:
- `cellpose.models.CellposeModel`: Initialized Cellpose cyto model

### `segment_cells(img_np)`

**Purpose**: Run Cellpose segmentation on a numpy image array.

**Args**:
- `img_np` (np.ndarray): RGB image array, shape (H, W, 3), dtype uint8

**Returns**:
- `np.ndarray` or `None`: Segmentation masks (0=background, >0=cell labels) or None if failed

### `masks_to_bboxes(masks)`

**Purpose**: Convert segmentation masks to bounding boxes.

**Args**:
- `masks` (np.ndarray): Segmentation mask array from Cellpose

**Returns**:
- `list` of `list`: Bounding boxes in `[x1, y1, x2, y2]` format, one per detected cell

### `save_coco(image_name, bboxes, img_width, img_height, image_id=1)`

**Purpose**: Save annotations in MS COCO format.

**Args**:
- `image_name` (str): Filename of the source image
- `bboxes` (list): List of bounding boxes `[x1, y1, x2, y2]`
- `img_width` (int): Width of the source image in pixels
- `img_height` (int): Height of the source image in pixels
- `image_id` (int): Unique identifier for this image (default: 1)

**Returns**:
- `dict`: COCO-formatted dictionary with keys: `images`, `annotations`, `categories`

### `main()`

**Purpose**: Main execution function. Processes all WSIs in the input directory.

**Command line usage**:
```bash
python cell_segmentation_to_coco.py <path_to_images_directory>
```

**Args**: None (reads from `sys.argv`)

**Returns**: None (saves JSON files to `./annotations/cellpose_annotations/`)

**Output files**: `{image_name}_wsi_coco.json` per processed WSI

## 2. train_test_split.py

### `copy_images_to_folders(image_source_dir, train_slides, val_slides, test_slides)`

**Purpose**: Copy WSI images from source directory to split folders.

**Args**:
- `image_source_dir` (str): Path to directory containing original WSI `.tiff` files
- `train_slides` (list): List of slide IDs for training set
- `val_slides` (list): List of slide IDs for validation set
- `test_slides` (list): List of slide IDs for test set

**Returns**:
- `tuple` of `(int, int, int)`: (train_count, val_count, test_count) - number of images copied to each folder

### `main()` (implicit in script execution)

**Purpose**: Stratified 80/10/10 split by domain (Tumor x Scanner x Origin x Species).

**Command line usage**:
```bash
python train_test_split.py <path_to_images_directory>
```

**Input files expected**:
- `../datasets_xvalidation.csv` (relative path) - metadata CSV with semicolon separator
- Image files in `<path_to_images_directory>`

**Output files**:
- `train.csv`, `val.csv`, `test.csv` - CSV files with slide metadata
- `images_split/train/`, `images_split/val/`, `images_split/test/` - Copied WSI images

**Returns**: None

## 3. 224_patch_around_bbox.py

### `extract_patches_224(coco_json, image_dir, output_dir, patch_size=224)`

**Purpose**: Extract 224x224 patches centered on each COCO bounding box.

**Args**:
- `coco_json` (str): Path to COCO JSON annotation file
- `image_dir` (str): Directory containing source WSI images
- `output_dir` (str): Directory to save extracted patches
- `patch_size` (int): Size of output patches in pixels (default: 224)

**Returns**:
- `list` of `dict`: Patch metadata list containing:
  - `patch_name` (str): Filename of extracted patch
  - `image_id` (int): Original image ID from COCO
  - `annotation_id` (int): Annotation ID from COCO
  - `category_id` (int): Category ID (1 or 2)
  - `category_name` (str): "mitotic" or "non-mitotic"
  - `original_bbox` (list): Original `[x1, y1, x2, y2]` coordinates
  - `patch_coords` (list): Patch crop `[left, top, right, bottom]` coordinates

**Output files**:
- Individual patch images: `{image_name}_ann{index}.tif`
- `patch_metadata.json` in output directory

### `main()` (implicit)

**Purpose**: Process train, val, and test splits sequentially.

**Command line usage**:
```bash
python 224_patch_around_bbox.py <path_to_coco_json>
```

**Expected directory structure**:
- `./images_split/train/` - Training images
- `./images_split/val/` - Validation images
- `./images_split/test/` - Test images

**Output directories**:
- `./images_split/train/224_patches/`
- `./images_split/val/224_patches/`
- `./images_split/test/224_patches/`

**Returns**: None

## 4. control_run_cnn.py

### `MitosisDataset.__init__(metadata_path, patches_dir)`

**Purpose**: Initialize dataset for baseline CNN.

**Args**:
- `metadata_path` (str): Path to `patch_metadata.json`
- `patches_dir` (str): Directory containing patch images

**Returns**: None

**Attributes created**:
- `self.metadata` (list): Loaded patch metadata
- `self.patches_dir` (str): Path to patches
- `self.transform` (Compose): ToTensor + ImageNet normalization
- `self.cat_to_label` (dict): {1: 1, 2: 0}

### `MitosisDataset.__getitem__(idx)`

**Purpose**: Get image and label at index.

**Args**:
- `idx` (int): Index into dataset

**Returns**:
- `tuple`: `(image_tensor, label)` where:
  - `image_tensor` (torch.Tensor): Shape (3, 224, 224), normalized
  - `label` (int): 1 for mitotic, 0 for non-mitotic

### `SimpleCNN.__init__(num_classes=2)`

**Purpose**: Initialize ResNet50 backbone with new classification head.

**Args**:
- `num_classes` (int): Number of output classes (default: 2)

**Returns**: None

### `SimpleCNN.forward(x)`

**Purpose**: Forward pass through the network.

**Args**:
- `x` (torch.Tensor): Input tensor, shape (B, 3, 224, 224)

**Returns**:
- `torch.Tensor`: Logits, shape (B, num_classes)

### `train_one_epoch(model, loader, loss_fn, optimizer, device)`

**Purpose**: Train model for one epoch.

**Args**:
- `model` (SimpleCNN): Neural network model
- `loader` (DataLoader): Training data loader
- `loss_fn` (nn.Module): Loss function (CrossEntropyLoss)
- `optimizer` (torch.optim): Optimizer (Adam)
- `device` (torch.device): 'cuda' or 'cpu'

**Returns**:
- `tuple`: `(avg_loss, accuracy)` where:
  - `avg_loss` (float): Average loss over epoch
  - `accuracy` (float): Training accuracy percentage

### `evaluate(model, loader, loss_fn, device)`

**Purpose**: Evaluate model on validation/test set.

**Args**:
- `model` (SimpleCNN): Neural network model
- `loader` (DataLoader): Evaluation data loader
- `loss_fn` (nn.Module): Loss function
- `device` (torch.device): 'cuda' or 'cpu'

**Returns**:
- `tuple`: `(avg_loss, accuracy, all_preds, all_labels)` where:
  - `avg_loss` (float): Average loss
  - `accuracy` (float): Accuracy percentage
  - `all_preds` (list): Predicted labels
  - `all_labels` (list): Ground truth labels

### `plot_training_history(train_losses, train_accs, test_losses, test_accs)`

**Purpose**: Plot and save training/validation curves.

**Args**:
- `train_losses` (list): Training loss per epoch
- `train_accs` (list): Training accuracy per epoch
- `test_losses` (list): Validation loss per epoch
- `test_accs` (list): Validation accuracy per epoch

**Returns**: None (saves `training_history.png`)

### `main()`

**Purpose**: Main training pipeline for baseline CNN.

**Output files**:
- `best_model.pth` - Best model weights
- `training_history.png` - Loss/accuracy curves

**Returns**: None

## 5. control_run_cnn_deconvolution.py

**Same functions as `control_run_cnn.py` with these additions:**

### `MitosisDataset.__init__(metadata_path, patches_dir, split='train')`

**Additional behavior**:
- Converts RGB to hematoxylin channel via HistomicsTK color deconvolution
- Applies robust percentile normalization [1%, 99%]
- Inverts color (1.0 - h_img) for ResNet compatibility
- Repeats single channel to 3 channels for ResNet50 input

**Args**:
- `split` (str): 'train', 'val', or 'test' - controls augmentation

## 6. dann_w_image_aug.py

### `GradientReversalFunction.forward(ctx, x, lambda_val)`

**Purpose**: Identity function in forward pass.

**Args**:
- `ctx` (context): Context for saving tensors
- `x` (torch.Tensor): Input tensor
- `lambda_val` (float): GRL strength parameter

**Returns**: `x.clone()` - identical tensor

### `GradientReversalFunction.backward(ctx, grad_output)`

**Purpose**: Reverse and scale gradients.

**Args**:
- `ctx` (context): Contains saved lambda_val
- `grad_output` (torch.Tensor): Gradient from upper layers

**Returns**: `-lambda_val * grad_output, None`

### `GradientReversalLayer.forward(x)`

**Purpose**: Apply gradient reversal.

**Args**:
- `x` (torch.Tensor): Input tensor

**Returns**: Reversed tensor

### `MitosisDataset.__init__(metadata_path, patches_dir, csv_path, is_train=True)`

**Purpose**: Initialize DANN dataset with domain labels.

**Args**:
- `metadata_path` (str): Path to `patch_metadata.json`
- `patches_dir` (str): Directory containing patch images
- `csv_path` (str): Path to train/val/test.csv with domain columns
- `is_train` (bool): If True, applies augmentation transforms

**Returns**: None

**Attributes**:
- `self.domain_maps` (dict): Maps domain values to integer indices
- `self.num_domain_classes` (dict): Number of classes per domain

### `MitosisDataset.__getitem__(idx)`

**Purpose**: Get image, mitosis label, and domain labels.

**Returns**:
- `tuple`: `(image_tensor, mitosis_label, domain_labels_dict)` where:
  - `image_tensor` (torch.Tensor): Shape (3, 224, 224)
  - `mitosis_label` (int): 1 for mitotic, 0 for non-mitotic
  - `domain_labels_dict` (dict): `{'Tumor': int, 'Species': int, 'Origin': int, 'Scanner': int}`

### `collate_fn(batch)`

**Purpose**: Custom collate for batched domain labels.

**Args**:
- `batch` (list): List of tuples from `__getitem__`

**Returns**:
- `tuple`: `(images, labels, domain_labels)` where:
  - `images` (torch.Tensor): Stacked images, shape (B, 3, 224, 224)
  - `labels` (torch.Tensor): Mitosis labels, shape (B,)
  - `domain_labels` (dict): Dict of tensors per attribute, shape (B,)

### `DANNModel.__init__(num_classes=2, num_domain_classes=None, lambda_val=0.0)`

**Purpose**: Initialize multi-domain DANN model.

**Args**:
- `num_classes` (int): Binary classification (default: 2)
- `num_domain_classes` (dict): `{'Tumor': int, 'Species': int, 'Origin': int, 'Scanner': int}`
- `lambda_val` (float): Initial GRL strength (default: 0.0)

**Returns**: None

### `DANNModel.forward(x)`

**Purpose**: Forward pass through feature extractor and all heads.

**Args**:
- `x` (torch.Tensor): Input tensor, shape (B, 3, 224, 224)

**Returns**:
- `tuple`: `(mitosis_logits, domain_logits)` where:
  - `mitosis_logits` (torch.Tensor): Shape (B, 2)
  - `domain_logits` (dict): Dict of tensors per attribute

### `DANNModel.set_lambda(val)`

**Purpose**: Update GRL strength for all domain adapters.

**Args**:
- `val` (float): New lambda value

**Returns**: None

### `DANNModel.predict_only(x)`

**Purpose**: Inference with only mitosis head (skip domain heads).

**Args**:
- `x` (torch.Tensor): Input tensor

**Returns**: `mitosis_logits` (torch.Tensor), shape (B, 2)

### `get_lambda(epoch, total_epochs, lambda_max=1.0)`

**Purpose**: Compute lambda schedule from DANN paper.

**Args**:
- `epoch` (int): Current epoch (0-indexed)
- `total_epochs` (int): Total training epochs
- `lambda_max` (float): Maximum lambda value (default: 1.0)

**Returns**:
- `float`: Lambda value for this epoch (ramps from 0 to lambda_max)

**Formula**: `lambda_max * (2.0 / (1.0 + exp(-10 * epoch/total_epochs)) - 1.0)`

### `train_one_epoch(model, loader, mitosis_loss_fn, domain_loss_fn, optimizer, device, lambda_val)`

**Purpose**: Train DANN for one epoch with adaptive weighting.

**Args**:
- `model` (DANNModel): Multi-domain DANN model
- `loader` (DataLoader): Training data loader
- `mitosis_loss_fn` (nn.Module): CrossEntropyLoss for mitosis (with class weights)
- `domain_loss_fn` (nn.Module): CrossEntropyLoss for domains (unweighted)
- `optimizer` (torch.optim): Adam optimizer with differential LRs
- `device` (torch.device): 'cuda' or 'cpu'
- `lambda_val` (float): Current GRL strength

**Returns**:
- `tuple`: `(avg_loss, avg_mitosis_loss, avg_domain_loss, accuracy)`

**Adaptive weight formula**: `adaptive_weight = running_mitosis_loss / (running_domain_loss + 1e-8)`

**Final loss**: `loss = loss_mitosis + adaptive_weight * loss_domains`

### `evaluate(model, loader, mitosis_loss_fn, domain_loss_fn, device)`

**Purpose**: Evaluate DANN on validation/test set.

**Args**:
- Same as `train_one_epoch` (without optimizer and lambda_val)

**Returns**:
- `tuple`: `(avg_loss, mitosis_acc, domain_accs, all_preds, all_labels)` where:
  - `avg_loss` (float): Average combined loss
  - `mitosis_acc` (float): Mitosis accuracy percentage
  - `domain_accs` (dict): Per-domain accuracy percentages
  - `all_preds` (list): Predicted mitosis labels
  - `all_labels` (list): Ground truth mitosis labels

### `plot_training_history(history, chance_levels)`

**Purpose**: Plot 6-panel training history.

**Args**:
- `history` (dict): Dictionary with keys: 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_domain_accs', 'test_domain_accs'
- `chance_levels` (dict): Per-domain chance levels (100 / num_classes)

**Returns**: None (saves `training_history.png`)

### `extract_features(model, dataloader, device)`

**Purpose**: Extract 2048-d features for UMAP visualization.

**Args**:
- `model` (DANNModel): Trained model
- `dataloader` (DataLoader): Test data loader
- `device` (torch.device): 'cuda' or 'cpu'

**Returns**:
- `tuple`: `(features, labels, domain_labels)` where:
  - `features` (np.ndarray): Shape (N, 2048)
  - `labels` (np.ndarray): Mitosis labels
  - `domain_labels` (dict): Per-domain label arrays

### `plot_umap(features, labels, all_domain_labels, dataset, save_path)`

**Purpose**: Create 5-panel UMAP visualization.

**Args**:
- `features` (np.ndarray): Feature matrix, shape (N, 2048)
- `labels` (np.ndarray): Mitosis labels
- `all_domain_labels` (dict): Per-domain label arrays
- `dataset` (MitosisDataset): Dataset with domain maps for legend labels
- `save_path` (str): Where to save the figure

**Returns**: None (saves UMAP plot)

### `main()`

**Purpose**: Complete DANN training pipeline.

**Output files** (saved to current directory):
- `best_dann_model.pth` - Best model weights
- `training_history.png` - 6-panel training curves
- `umap_dann.png` - UMAP visualization

**Returns**: None

## 7. dann_deconvolution.py

**Similar to `dann_w_image_aug.py` with these differences:**

### `MitosisDataset.__getitem__(idx)`

**Key difference**: Converts RGB to hematoxylin channel (1-channel input)

**Returns**:
- `image` (torch.Tensor): Shape (1, 224, 224) - single channel
- (model internally converts to 1-channel input for ResNet)

### `DANNModel.__init__()`

**Key difference**: First conv layer modified to accept 1-channel input

**Behavior**: 
- Loads standard ResNet50
- Replaces `conv1` (3-channel) with 1-channel version
- Copies weights by averaging across original RGB channels

## 8. final_model.py

**Similar to `dann_w_image_aug.py` with these additions:**

### `ShotNoise.__init__(scale=0.05)`

**Purpose**: Simulate photon counting noise (targets Scanner domain).

**Args**:
- `scale` (float): Noise magnitude (smaller = more photons = finer grain)

**Returns**: None (callable)

### `ShotNoise.__call__(img)`

**Args**:
- `img` (PIL.Image): Input image

**Returns**: `PIL.Image` with Poisson noise applied

### `DefocusBlur.__init__(kernel_size=9, sigma_low=1.5, sigma_high=4.0)`

**Purpose**: Simulate out-of-focus lens aberrations.

**Args**:
- `kernel_size` (int): Gaussian blur kernel size
- `sigma_low` (float): Minimum sigma
- `sigma_high` (float): Maximum sigma

**Returns**: None (callable)

### `DANNModel.__init__()`

**Key difference**: Multi-scale feature extraction

**Architecture**:
- `self.stem` = conv1 + bn1 + relu + maxpool + layer1
- `self.layer2`, `self.layer3`, `self.layer4` = ResNet stages
- `self.pool2`, `self.pool3`, `self.pool4` = AdaptiveAvgPool2d(1)
- Feature dimension = 512 + 1024 + 2048 = **3584**

### `DANNModel.get_features(x)`

**Purpose**: Extract multi-scale features for UMAP.

**Args**:
- `x` (torch.Tensor): Input tensor

**Returns**:
- `torch.Tensor`: Concatenated features [B, 3584]

### `plot_auc(all_labels, all_probs, save_path)`

**Purpose**: Generate ROC curve and compute AUC.

**Args**:
- `all_labels` (list): Ground truth labels (0/1)
- `all_probs` (list): Softmax probability of class 1 (mitotic)
- `save_path` (str): Output path for figure

**Returns**:
- `float`: ROC-AUC score

### `plot_f1_heatmap(all_preds, all_labels, all_domain_label_list, dataset, save_path)`

**Purpose**: Create per-tumor-type F1 heatmap.

**Args**:
- `all_preds` (list): Predicted labels
- `all_labels` (list): Ground truth labels
- `all_domain_label_list` (dict): Domain labels from evaluate()
- `dataset` (MitosisDataset): Dataset with Tumor domain map
- `save_path` (str): Output path for figure

**Returns**: None (saves F1 heatmap)

### `main()`

**Purpose**: Complete multi-stage DANN training with multi-scale features.

**Output files** (saved to `results/` directory):
- `best_dann_model.pth` - Best model weights
- `training_history.png` - 7-panel training curves
- `auc_curve.png` - ROC curve with AUC
- `f1_heatmap_tumor.png` - Per-tumor-type F1 scores
- `umap_dann.png` - 5-panel UMAP visualization

**Returns**: None

## Common Data Structures

### patch_metadata.json format

```json
[
  {
    "patch_name": "001_ann0.tif",
    "image_id": 1,
    "annotation_id": 123,
    "category_id": 1,
    "category_name": "mitotic",
    "original_bbox": [100, 200, 300, 400],
    "patch_coords": [88, 188, 312, 412]
  }
]
```

### CSV format (train.csv / val.csv / test.csv)

| Slide | Tumor | Species | Origin | Scanner |
|-------|-------|---------|--------|---------|
| 001 | human_breast_cancer | human | AMC | Hamamatsu XR |

### COCO format (for annotations)

```json
{
  "images": [{"id": 1, "file_name": "001.tif", "width": 7200, "height": 5400}],
  "annotations": [{"id": 1, "image_id": 1, "bbox": [x, y, w, h], "category_id": 1, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "cell"}]
}
```

Gemini was used to help format and draft the documentation.md based on my original code. I reviewed and edited all descriptions for technical accuracy.

