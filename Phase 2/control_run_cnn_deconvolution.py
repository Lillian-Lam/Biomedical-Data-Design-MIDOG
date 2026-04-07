import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Hematoxylin extraction via H&E color deconvolution (HistomicsTK)
# We import only the required submodule to avoid pulling optional heavy dependencies.
try:
    from histomicstk.preprocessing.color_deconvolution import color_deconvolution, stain_color_map
except ImportError as e:
    raise ImportError(
        "Failed to import HistomicsTK color deconvolution submodule. "
        "If you installed histomicstk with --no-deps, also install minimal deps "
        "(numpy/scipy/pillow/scikit-image). Original error: " + str(e)
    )

from tqdm import tqdm
import matplotlib.pyplot as plt

# File paths - MAKE SURE THESE MATCH YOUR DIRECTORY STRUCTURE
train_path = './images_split/train/224_patches'
val_path = './images_split/val/224_patches'
test_path = './images_split/test/224_patches/'
train_metadata = os.path.join(train_path, 'patch_metadata.json')
val_metadata = os.path.join(val_path, 'patch_metadata.json')
test_metadata = os.path.join(test_path, 'patch_metadata.json')

# Training hyperparameters
num_epochs = 20
batch_size = 32
LEARNING_RATE = 1e-4       # Lower learning rate for fine-tuning the pre-trained backbone
LR_STEP_SIZE = 5           # Step size for StepLR
LR_GAMMA = 0.5             # Gamma for StepLR

class MitosisDataset(Dataset):
    def __init__(self, metadata_path, patches_dir, split: str = 'train'):
        self.split = split
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.patches_dir = patches_dir
        
        self.to_tensor = transforms.ToTensor()
        
        # [FIX] Use standard ImageNet normalization since we are mimicking a 3-channel input
        self.normalize_h = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Pre-build stain matrix W for H&E (hematoxylin, eosin, null)
        stains = ['hematoxylin', 'eosin', 'null']
        self.W = np.array([stain_color_map[st] for st in stains]).T
        self.transform = None

        # Only apply geometric augmentations during training 
        # (ColorJitter is disabled as it interferes with HistomicsTK stain deconvolution)
        if self.split == 'train':
            self.use_aug = True
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(90)], p=0.8)
            ])
        else:
            self.use_aug = False

        # mitotic(1) -> 1, non-mitotic(2) -> 0
        self.cat_to_label = {1: 1, 2: 0}
        
        mitotic = sum(1 for item in self.metadata if item['category_id'] == 1)
        non_mitotic = sum(1 for item in self.metadata if item['category_id'] == 2)
        print(f'Loaded {len(self.metadata)} patches ({mitotic} mitotic, {non_mitotic} non-mitotic)')

    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.patches_dir, item['patch_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation for the training split only
        if self.use_aug and hasattr(self, 'aug_transform'):
            image = self.aug_transform(image)

        # Convert PIL image to numpy array for color deconvolution
        if isinstance(image, Image.Image):
            rgb = np.array(image, dtype=np.uint8)[:, :, :3]
        else:
            tensor = image
            rgb = (tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

        # Perform standard color deconvolution with a fixed H&E stain matrix
        im_deconv = color_deconvolution(rgb, self.W)
        
        # Extract the Hematoxylin channel (stain density)
        h_img = im_deconv.StainsFloat[:, :, 0].astype(np.float32)

        # Robust normalization to [0, 1] using percentiles to handle scanner brightness variations
        p1, p99 = np.percentile(h_img, [1, 99])
        if p99 > p1:
            h_img = (h_img - p1) / (p99 - p1)
        else:
            h_img = h_img - p1
        h_img = np.clip(h_img, 0.0, 1.0)

        # [CRITICAL FIX 1] Color Inversion
        # Optical Density is high (white/1.0) for nuclei and low (black/0.0) for background.
        # ResNet pre-trained filters expect dark edges on a light background.
        # We invert the colors to match natural image characteristics.
        h_img = 1.0 - h_img

        # [CRITICAL FIX 2] Convert the single channel into a 3-channel tensor
        # This perfectly fits the standard ResNet50 input requirements without altering conv1.
        h = torch.from_numpy(h_img.astype(np.float32)).unsqueeze(0)
        h = h.repeat(3, 1, 1) 
        
        # Apply ImageNet normalization
        image = self.normalize_h(h).float()
        
        label = self.cat_to_label[item['category_id']]
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Use standard pre-trained ResNet50 without modifying the base architecture
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        # Replace the final fully connected layer for binary classification
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar= tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs= model(images)
        loss= loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss+= loss.item()
        _, predicted= outputs.max(1)
        total+= labels.size(0)
        correct+= predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss/len(loader), 100. * (correct/total)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * (correct/total)
    avg_loss = running_loss/len(loader)
    
    return avg_loss, accuracy, all_preds, all_labels

def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot the accuracy 
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n Training history plot saved as 'training_history.png'")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('\n Loading datasets...')
    train_dataset = MitosisDataset(train_metadata, train_path, split='train')
    val_dataset = MitosisDataset(val_metadata, val_path, split='val')
    test_dataset = MitosisDataset(test_metadata, test_path, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print('\n Initializing model...')
    model = SimpleCNN(num_classes=2).to(device)
    
    # Calculate class weights dynamically to address dataset imbalance
    mitotic_count = sum(1 for item in train_dataset.metadata if item['category_id'] == 1)
    non_mitotic_count = sum(1 for item in train_dataset.metadata if item['category_id'] == 2)
    total = mitotic_count + non_mitotic_count
    
    weight_non_mitotic = total / (2.0 * non_mitotic_count) # Weight for label 0
    weight_mitotic = total / (2.0 * mitotic_count)         # Weight for label 1
    class_weights = torch.tensor([weight_non_mitotic, weight_mitotic]).to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # [CRITICAL FIX 3] Train end-to-end from the beginning (No Backbone Freezing)
    # The domain shift from ImageNet to Pathology is large; fine-tuning the whole network immediately is necessary.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    # Data storage for plotting
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f'\n Starting training for {num_epochs} epochs...')
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training step
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Validation step
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(val_loss)
        test_accs.append(val_acc)
        
        print(f'\n Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f' Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f' Saved best model (val_acc: {val_acc:.2f}%)')

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f' Current LR: {current_lr:.6g}')
    
    print('\n Plotting training history...')
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    print(" Final evaluation on the test set")
    # Load the best weights before final testing
    model.load_state_dict(torch.load('best_model.pth'))
    _, _, all_preds, all_labels = evaluate(model, test_loader, loss_fn, device)
    
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, 
                               target_names=['Non-mitotic', 'Mitotic']))
    
    cm = confusion_matrix(all_labels, all_preds)
    print('\nConfusion Matrix:')
    print(cm)
    
    # Calculate specific metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f'\n Key Metrics:')
    print(f'     Sensitivity (Recall): {sensitivity:.3f}')
    print(f'     Specificity: {specificity:.3f}')
    print(f'     Precision: {precision:.3f}')
    print(f'     F1-Score: {f1:.3f}')
    print(f'     Best Val Accuracy: {best_val_acc:.2f}%')

if __name__ == "__main__":
    main()