\# Phase 2 - Function Documentation



\## Overview

This document provides detailed function-level documentation for all Phase 2 scripts, including input/output expectations, argument types, and return values.



\## 1. cell\_segmentation\_to\_coco.py



\### `get\_wsi\_files(wsi\_dir)`

&#x20; \*\*Purpose\*\*: Get all TIFF WSI files from a directory.

&#x20; 

&#x20; \*\*Args\*\*:

&#x20; - `wsi\_dir` (str): Path to directory containing WSI files

&#x20; 

&#x20; \*\*Returns\*\*:

&#x20; - `list` of str: Full paths to all `.tif` and `.tiff` files in the directory



\### `get\_cellpose\_model()`

&#x20; \*\*Purpose\*\*: Get or initialize the global Cellpose model instance (singleton pattern).

&#x20; 

&#x20; \*\*Args\*\*: None

&#x20; 

&#x20; \*\*Returns\*\*:

&#x20; - `cellpose.models.CellposeModel`: Initialized Cellpose cyto model





\### `segment\_cells(img\_np)`

\*\*Purpose\*\*: Run Cellpose segmentation on a numpy image array.



\*\*Args\*\*:

\- `img\_np` (np.ndarray): RGB image array, shape (H, W, 3), dtype uint8



\*\*Returns\*\*:

\- `np.ndarray` or `None`: Segmentation masks (0=background, >0=cell labels) or None if failed



\### `masks\_to\_bboxes(masks)`

\*\*Purpose\*\*: Convert segmentation masks to bounding boxes.



\*\*Args\*\*:

\- `masks` (np.ndarray): Segmentation mask array from Cellpose



\*\*Returns\*\*:

\- `list` of `list`: Bounding boxes in `\[x1, y1, x2, y2]` format, one per detected cell



\### `save\_coco(image\_name, bboxes, img\_width, img\_height, image\_id=1)`

\*\*Purpose\*\*: Save annotations in MS COCO format.



\*\*Args\*\*:

\- `image\_name` (str): Filename of the source image

\- `bboxes` (list): List of bounding boxes `\[x1, y1, x2, y2]`

\- `img\_width` (int): Width of the source image in pixels

\- `img\_height` (int): Height of the source image in pixels

\- `image\_id` (int): Unique identifier for this image (default: 1)



\*\*Returns\*\*:

\- `dict`: COCO-formatted dictionary with keys: `images`, `annotations`, `categories`



\### `main()`

\*\*Purpose\*\*: Main execution function. Processes all WSIs in the input directory.



\*\*Command line usage\*\*:

```bash

python cell\_segmentation\_to\_coco.py <path\_to\_images\_directory>

```



\*\*Args\*\*: None (reads from `sys.argv`)



\*\*Returns\*\*: None (saves JSON files to `./annotations/cellpose\_annotations/`)



\*\*Output files\*\*: `{image\_name}\_wsi\_coco.json` per processed WSI





\## 2. train\_test\_split.py



\### `copy\_images\_to\_folders(image\_source\_dir, train\_slides, val\_slides, test\_slides)`

\*\*Purpose\*\*: Copy WSI images from source directory to split folders.



\*\*Args\*\*:

\- `image\_source\_dir` (str): Path to directory containing original WSI `.tiff` files

\- `train\_slides` (list): List of slide IDs for training set

\- `val\_slides` (list): List of slide IDs for validation set

\- `test\_slides` (list): List of slide IDs for test set



\*\*Returns\*\*:

\- `tuple` of `(int, int, int)`: (train\_count, val\_count, test\_count) - number of images copied to each folder



\### `main()` (implicit in script execution)

\*\*Purpose\*\*: Stratified 80/10/10 split by domain (Tumor x Scanner x Origin x Species).



\*\*Command line usage\*\*:

```bash

python train\_test\_split.py <path\_to\_images\_directory>

```



\*\*Input files expected\*\*:

\- `../datasets\_xvalidation.csv` (relative path) - metadata CSV with semicolon separator

\- Image files in `<path\_to\_images\_directory>`



\*\*Output files\*\*:

\- `train.csv`, `val.csv`, `test.csv` - CSV files with slide metadata

\- `images\_split/train/`, `images\_split/val/`, `images\_split/test/` - Copied WSI images



\*\*Returns\*\*: None





\## 3. 224\_patch\_around\_bbox.py



\### `extract\_patches\_224(coco\_json, image\_dir, output\_dir, patch\_size=224)`

\*\*Purpose\*\*: Extract 224x224 patches centered on each COCO bounding box.



\*\*Args\*\*:

\- `coco\_json` (str): Path to COCO JSON annotation file

\- `image\_dir` (str): Directory containing source WSI images

\- `output\_dir` (str): Directory to save extracted patches

\- `patch\_size` (int): Size of output patches in pixels (default: 224)



\*\*Returns\*\*:

\- `list` of `dict`: Patch metadata list containing:

&#x20; - `patch\_name` (str): Filename of extracted patch

&#x20; - `image\_id` (int): Original image ID from COCO

&#x20; - `annotation\_id` (int): Annotation ID from COCO

&#x20; - `category\_id` (int): Category ID (1 or 2)

&#x20; - `category\_name` (str): "mitotic" or "non-mitotic"

&#x20; - `original\_bbox` (list): Original `\[x1, y1, x2, y2]` coordinates

&#x20; - `patch\_coords` (list): Patch crop `\[left, top, right, bottom]` coordinates



\*\*Output files\*\*:

\- Individual patch images: `{image\_name}\_ann{index}.tif`

\- `patch\_metadata.json` in output directory



\### `main()` (implicit)

\*\*Purpose\*\*: Process train, val, and test splits sequentially.



\*\*Command line usage\*\*:

```bash

python 224\_patch\_around\_bbox.py <path\_to\_coco\_json>

```



\*\*Expected directory structure\*\*:

\- `./images\_split/train/` - Training images

\- `./images\_split/val/` - Validation images

\- `./images\_split/test/` - Test images



\*\*Output directories\*\*:

\- `./images\_split/train/224\_patches/`

\- `./images\_split/val/224\_patches/`

\- `./images\_split/test/224\_patches/`



\*\*Returns\*\*: None



\## 4. control\_run\_cnn.py



\### `MitosisDataset.\_\_init\_\_(metadata\_path, patches\_dir)`

\*\*Purpose\*\*: Initialize dataset for baseline CNN.



\*\*Args\*\*:

\- `metadata\_path` (str): Path to `patch\_metadata.json`

\- `patches\_dir` (str): Directory containing patch images



\*\*Returns\*\*: None



\*\*Attributes created\*\*:

\- `self.metadata` (list): Loaded patch metadata

\- `self.patches\_dir` (str): Path to patches

\- `self.transform` (Compose): ToTensor + ImageNet normalization

\- `self.cat\_to\_label` (dict): {1: 1, 2: 0}



\### `MitosisDataset.\_\_getitem\_\_(idx)`

\*\*Purpose\*\*: Get image and label at index.



\*\*Args\*\*:

\- `idx` (int): Index into dataset



\*\*Returns\*\*:

\- `tuple`: `(image\_tensor, label)` where:

&#x20; - `image\_tensor` (torch.Tensor): Shape (3, 224, 224), normalized

&#x20; - `label` (int): 1 for mitotic, 0 for non-mitotic



\### `SimpleCNN.\_\_init\_\_(num\_classes=2)`

\*\*Purpose\*\*: Initialize ResNet50 backbone with new classification head.



\*\*Args\*\*:

\- `num\_classes` (int): Number of output classes (default: 2)



\*\*Returns\*\*: None



\### `SimpleCNN.forward(x)`

\*\*Purpose\*\*: Forward pass through the network.



\*\*Args\*\*:

\- `x` (torch.Tensor): Input tensor, shape (B, 3, 224, 224)



\*\*Returns\*\*:

\- `torch.Tensor`: Logits, shape (B, num\_classes)



\### `train\_one\_epoch(model, loader, loss\_fn, optimizer, device)`

\*\*Purpose\*\*: Train model for one epoch.



\*\*Args\*\*:

\- `model` (SimpleCNN): Neural network model

\- `loader` (DataLoader): Training data loader

\- `loss\_fn` (nn.Module): Loss function (CrossEntropyLoss)

\- `optimizer` (torch.optim): Optimizer (Adam)

\- `device` (torch.device): 'cuda' or 'cpu'



\*\*Returns\*\*:

\- `tuple`: `(avg\_loss, accuracy)` where:

&#x20; - `avg\_loss` (float): Average loss over epoch

&#x20; - `accuracy` (float): Training accuracy percentage



\### `evaluate(model, loader, loss\_fn, device)`

\*\*Purpose\*\*: Evaluate model on validation/test set.



\*\*Args\*\*:

\- `model` (SimpleCNN): Neural network model

\- `loader` (DataLoader): Evaluation data loader

\- `loss\_fn` (nn.Module): Loss function

\- `device` (torch.device): 'cuda' or 'cpu'



\*\*Returns\*\*:

\- `tuple`: `(avg\_loss, accuracy, all\_preds, all\_labels)` where:

&#x20; - `avg\_loss` (float): Average loss

&#x20; - `accuracy` (float): Accuracy percentage

&#x20; - `all\_preds` (list): Predicted labels

&#x20; - `all\_labels` (list): Ground truth labels



\### `plot\_training\_history(train\_losses, train\_accs, test\_losses, test\_accs)`

\*\*Purpose\*\*: Plot and save training/validation curves.



\*\*Args\*\*:

\- `train\_losses` (list): Training loss per epoch

\- `train\_accs` (list): Training accuracy per epoch

\- `test\_losses` (list): Validation loss per epoch

\- `test\_accs` (list): Validation accuracy per epoch



\*\*Returns\*\*: None (saves `training\_history.png`)



\### `main()`

\*\*Purpose\*\*: Main training pipeline for baseline CNN.



\*\*Output files\*\*:

\- `best\_model.pth` - Best model weights

\- `training\_history.png` - Loss/accuracy curves



\*\*Returns\*\*: None



\## 5. control\_run\_cnn\_deconvolution.py



\*\*Same functions as `control\_run\_cnn.py` with these additions:\*\*



\### `MitosisDataset.\_\_init\_\_(metadata\_path, patches\_dir, split='train')`

\*\*Additional behavior\*\*:

\- Converts RGB to hematoxylin channel via HistomicsTK color deconvolution

\- Applies robust percentile normalization \[1%, 99%]

\- Inverts color (1.0 - h\_img) for ResNet compatibility

\- Repeats single channel to 3 channels for ResNet50 input



\*\*Args\*\*:

\- `split` (str): 'train', 'val', or 'test' - controls augmentation



\## 6. dann\_w\_image\_aug.py



\### `GradientReversalFunction.forward(ctx, x, lambda\_val)`

\*\*Purpose\*\*: Identity function in forward pass.



\*\*Args\*\*:

\- `ctx` (context): Context for saving tensors

\- `x` (torch.Tensor): Input tensor

\- `lambda\_val` (float): GRL strength parameter



\*\*Returns\*\*: `x.clone()` - identical tensor



\### `GradientReversalFunction.backward(ctx, grad\_output)`

\*\*Purpose\*\*: Reverse and scale gradients.



\*\*Args\*\*:

\- `ctx` (context): Contains saved lambda\_val

\- `grad\_output` (torch.Tensor): Gradient from upper layers



\*\*Returns\*\*: `-lambda\_val \* grad\_output, None`



\### `GradientReversalLayer.forward(x)`

\*\*Purpose\*\*: Apply gradient reversal.



\*\*Args\*\*:

\- `x` (torch.Tensor): Input tensor



\*\*Returns\*\*: Reversed tensor



\### `MitosisDataset.\_\_init\_\_(metadata\_path, patches\_dir, csv\_path, is\_train=True)`

\*\*Purpose\*\*: Initialize DANN dataset with domain labels.



\*\*Args\*\*:

\- `metadata\_path` (str): Path to `patch\_metadata.json`

\- `patches\_dir` (str): Directory containing patch images

\- `csv\_path` (str): Path to train/val/test.csv with domain columns

\- `is\_train` (bool): If True, applies augmentation transforms



\*\*Returns\*\*: None



\*\*Attributes\*\*:

\- `self.domain\_maps` (dict): Maps domain values to integer indices

\- `self.num\_domain\_classes` (dict): Number of classes per domain



\### `MitosisDataset.\_\_getitem\_\_(idx)`

\*\*Purpose\*\*: Get image, mitosis label, and domain labels.



\*\*Returns\*\*:

\- `tuple`: `(image\_tensor, mitosis\_label, domain\_labels\_dict)` where:

&#x20; - `image\_tensor` (torch.Tensor): Shape (3, 224, 224)

&#x20; - `mitosis\_label` (int): 1 for mitotic, 0 for non-mitotic

&#x20; - `domain\_labels\_dict` (dict): `{'Tumor': int, 'Species': int, 'Origin': int, 'Scanner': int}`



\### `collate\_fn(batch)`

\*\*Purpose\*\*: Custom collate for batched domain labels.



\*\*Args\*\*:

\- `batch` (list): List of tuples from `\_\_getitem\_\_`



\*\*Returns\*\*:

\- `tuple`: `(images, labels, domain\_labels)` where:

&#x20; - `images` (torch.Tensor): Stacked images, shape (B, 3, 224, 224)

&#x20; - `labels` (torch.Tensor): Mitosis labels, shape (B,)

&#x20; - `domain\_labels` (dict): Dict of tensors per attribute, shape (B,)



\### `DANNModel.\_\_init\_\_(num\_classes=2, num\_domain\_classes=None, lambda\_val=0.0)`

\*\*Purpose\*\*: Initialize multi-domain DANN model.



\*\*Args\*\*:

\- `num\_classes` (int): Binary classification (default: 2)

\- `num\_domain\_classes` (dict): `{'Tumor': int, 'Species': int, 'Origin': int, 'Scanner': int}`

\- `lambda\_val` (float): Initial GRL strength (default: 0.0)



\*\*Returns\*\*: None



\### `DANNModel.forward(x)`

\*\*Purpose\*\*: Forward pass through feature extractor and all heads.



\*\*Args\*\*:

\- `x` (torch.Tensor): Input tensor, shape (B, 3, 224, 224)



\*\*Returns\*\*:

\- `tuple`: `(mitosis\_logits, domain\_logits)` where:

&#x20; - `mitosis\_logits` (torch.Tensor): Shape (B, 2)

&#x20; - `domain\_logits` (dict): Dict of tensors per attribute



\### `DANNModel.set\_lambda(val)`

\*\*Purpose\*\*: Update GRL strength for all domain adapters.



\*\*Args\*\*:

\- `val` (float): New lambda value



\*\*Returns\*\*: None



\### `DANNModel.predict\_only(x)`

\*\*Purpose\*\*: Inference with only mitosis head (skip domain heads).



\*\*Args\*\*:

\- `x` (torch.Tensor): Input tensor



\*\*Returns\*\*: `mitosis\_logits` (torch.Tensor), shape (B, 2)



\### `get\_lambda(epoch, total\_epochs, lambda\_max=1.0)`

\*\*Purpose\*\*: Compute lambda schedule from DANN paper.



\*\*Args\*\*:

\- `epoch` (int): Current epoch (0-indexed)

\- `total\_epochs` (int): Total training epochs

\- `lambda\_max` (float): Maximum lambda value (default: 1.0)



\*\*Returns\*\*:

\- `float`: Lambda value for this epoch (ramps from 0 to lambda\_max)



\*\*Formula\*\*: `lambda\_max \* (2.0 / (1.0 + exp(-10 \* epoch/total\_epochs)) - 1.0)`



\### `train\_one\_epoch(model, loader, mitosis\_loss\_fn, domain\_loss\_fn, optimizer, device, lambda\_val)`

\*\*Purpose\*\*: Train DANN for one epoch with adaptive weighting.



\*\*Args\*\*:

\- `model` (DANNModel): Multi-domain DANN model

\- `loader` (DataLoader): Training data loader

\- `mitosis\_loss\_fn` (nn.Module): CrossEntropyLoss for mitosis (with class weights)

\- `domain\_loss\_fn` (nn.Module): CrossEntropyLoss for domains (unweighted)

\- `optimizer` (torch.optim): Adam optimizer with differential LRs

\- `device` (torch.device): 'cuda' or 'cpu'

\- `lambda\_val` (float): Current GRL strength



\*\*Returns\*\*:

\- `tuple`: `(avg\_loss, avg\_mitosis\_loss, avg\_domain\_loss, accuracy)`



\*\*Adaptive weight formula\*\*: `adaptive\_weight = running\_mitosis\_loss / (running\_domain\_loss + 1e-8)`



\*\*Final loss\*\*: `loss = loss\_mitosis + adaptive\_weight \* loss\_domains`



\### `evaluate(model, loader, mitosis\_loss\_fn, domain\_loss\_fn, device)`

\*\*Purpose\*\*: Evaluate DANN on validation/test set.



\*\*Args\*\*:

\- Same as `train\_one\_epoch` (without optimizer and lambda\_val)



\*\*Returns\*\*:

\- `tuple`: `(avg\_loss, mitosis\_acc, domain\_accs, all\_preds, all\_labels)` where:

&#x20; - `avg\_loss` (float): Average combined loss

&#x20; - `mitosis\_acc` (float): Mitosis accuracy percentage

&#x20; - `domain\_accs` (dict): Per-domain accuracy percentages

&#x20; - `all\_preds` (list): Predicted mitosis labels

&#x20; - `all\_labels` (list): Ground truth mitosis labels



\### `plot\_training\_history(history, chance\_levels)`

\*\*Purpose\*\*: Plot 6-panel training history.



\*\*Args\*\*:

\- `history` (dict): Dictionary with keys: 'train\_loss', 'test\_loss', 'train\_acc', 'test\_acc', 'train\_domain\_accs', 'test\_domain\_accs'

\- `chance\_levels` (dict): Per-domain chance levels (100 / num\_classes)



\*\*Returns\*\*: None (saves `training\_history.png`)



\### `extract\_features(model, dataloader, device)`

\*\*Purpose\*\*: Extract 2048-d features for UMAP visualization.



\*\*Args\*\*:

\- `model` (DANNModel): Trained model

\- `dataloader` (DataLoader): Test data loader

\- `device` (torch.device): 'cuda' or 'cpu'



\*\*Returns\*\*:

\- `tuple`: `(features, labels, domain\_labels)` where:

&#x20; - `features` (np.ndarray): Shape (N, 2048)

&#x20; - `labels` (np.ndarray): Mitosis labels

&#x20; - `domain\_labels` (dict): Per-domain label arrays



\### `plot\_umap(features, labels, all\_domain\_labels, dataset, save\_path)`

\*\*Purpose\*\*: Create 5-panel UMAP visualization.



\*\*Args\*\*:

\- `features` (np.ndarray): Feature matrix, shape (N, 2048)

\- `labels` (np.ndarray): Mitosis labels

\- `all\_domain\_labels` (dict): Per-domain label arrays

\- `dataset` (MitosisDataset): Dataset with domain maps for legend labels

\- `save\_path` (str): Where to save the figure



\*\*Returns\*\*: None (saves UMAP plot)



\### `main()`

\*\*Purpose\*\*: Complete DANN training pipeline.



\*\*Output files\*\* (saved to current directory):

\- `best\_dann\_model.pth` - Best model weights

\- `training\_history.png` - 6-panel training curves

\- `umap\_dann.png` - UMAP visualization



\*\*Returns\*\*: None



\## 7. dann\_deconvolution.py



\*\*Similar to `dann\_w\_image\_aug.py` with these differences:\*\*



\### `MitosisDataset.\_\_getitem\_\_(idx)`

\*\*Key difference\*\*: Converts RGB to hematoxylin channel (1-channel input)



\*\*Returns\*\*:

\- `image` (torch.Tensor): Shape (1, 224, 224) - single channel

\- (model internally converts to 1-channel input for ResNet)



\### `DANNModel.\_\_init\_\_()`

\*\*Key difference\*\*: First conv layer modified to accept 1-channel input



\*\*Behavior\*\*: 

\- Loads standard ResNet50

\- Replaces `conv1` (3-channel) with 1-channel version

\- Copies weights by averaging across original RGB channels



\## 8. final\_model.py



\*\*Similar to `dann\_w\_image\_aug.py` with these additions:\*\*



\### `ShotNoise.\_\_init\_\_(scale=0.05)`

\*\*Purpose\*\*: Simulate photon counting noise (targets Scanner domain).



\*\*Args\*\*:

\- `scale` (float): Noise magnitude (smaller = more photons = finer grain)



\*\*Returns\*\*: None (callable)



\### `ShotNoise.\_\_call\_\_(img)`

\*\*Args\*\*:

\- `img` (PIL.Image): Input image



\*\*Returns\*\*: `PIL.Image` with Poisson noise applied



\### `DefocusBlur.\_\_init\_\_(kernel\_size=9, sigma\_low=1.5, sigma\_high=4.0)`

\*\*Purpose\*\*: Simulate out-of-focus lens aberrations.



\*\*Args\*\*:

\- `kernel\_size` (int): Gaussian blur kernel size

\- `sigma\_low` (float): Minimum sigma

\- `sigma\_high` (float): Maximum sigma



\*\*Returns\*\*: None (callable)



\### `DANNModel.\_\_init\_\_()`

\*\*Key difference\*\*: Multi-scale feature extraction



\*\*Architecture\*\*:

\- `self.stem` = conv1 + bn1 + relu + maxpool + layer1

\- `self.layer2`, `self.layer3`, `self.layer4` = ResNet stages

\- `self.pool2`, `self.pool3`, `self.pool4` = AdaptiveAvgPool2d(1)

\- Feature dimension = 512 + 1024 + 2048 = \*\*3584\*\*



\### `DANNModel.get\_features(x)`

\*\*Purpose\*\*: Extract multi-scale features for UMAP.



\*\*Args\*\*:

\- `x` (torch.Tensor): Input tensor



\*\*Returns\*\*:

\- `torch.Tensor`: Concatenated features \[B, 3584]



\### `plot\_auc(all\_labels, all\_probs, save\_path)`

\*\*Purpose\*\*: Generate ROC curve and compute AUC.



\*\*Args\*\*:

\- `all\_labels` (list): Ground truth labels (0/1)

\- `all\_probs` (list): Softmax probability of class 1 (mitotic)

\- `save\_path` (str): Output path for figure



\*\*Returns\*\*:

\- `float`: ROC-AUC score



\### `plot\_f1\_heatmap(all\_preds, all\_labels, all\_domain\_label\_list, dataset, save\_path)`

\*\*Purpose\*\*: Create per-tumor-type F1 heatmap.



\*\*Args\*\*:

\- `all\_preds` (list): Predicted labels

\- `all\_labels` (list): Ground truth labels

\- `all\_domain\_label\_list` (dict): Domain labels from evaluate()

\- `dataset` (MitosisDataset): Dataset with Tumor domain map

\- `save\_path` (str): Output path for figure



\*\*Returns\*\*: None (saves F1 heatmap)



\### `main()`

\*\*Purpose\*\*: Complete multi-stage DANN training with multi-scale features.



\*\*Output files\*\* (saved to `results/` directory):

\- `best\_dann\_model.pth` - Best model weights

\- `training\_history.png` - 7-panel training curves

\- `auc\_curve.png` - ROC curve with AUC

\- `f1\_heatmap\_tumor.png` - Per-tumor-type F1 scores

\- `umap\_dann.png` - 5-panel UMAP visualization



\*\*Returns\*\*: None



\## Common Data Structures



\### patch\_metadata.json format

```json

\[

&#x20; {

&#x20;   "patch\_name": "001\_ann0.tif",

&#x20;   "image\_id": 1,

&#x20;   "annotation\_id": 123,

&#x20;   "category\_id": 1,

&#x20;   "category\_name": "mitotic",

&#x20;   "original\_bbox": \[100, 200, 300, 400],

&#x20;   "patch\_coords": \[88, 188, 312, 412]

&#x20; }

]

```



\### CSV format (train.csv / val.csv / test.csv)

| Slide | Tumor | Species | Origin | Scanner |

|-------|-------|---------|--------|---------|

| 001 | human\_breast\_cancer | human | AMC | Hamamatsu XR |



\### COCO format (for annotations)

```json

{

&#x20; "images": \[{"id": 1, "file\_name": "001.tif", "width": 7200, "height": 5400}],

&#x20; "annotations": \[{"id": 1, "image\_id": 1, "bbox": \[x, y, w, h], "category\_id": 1, "iscrowd": 0}],

&#x20; "categories": \[{"id": 1, "name": "cell"}]

}

```


Gemini was used to help format and draft the documentation.md based on my original code. I reviewed and edited all descriptions for technical accuracy.

