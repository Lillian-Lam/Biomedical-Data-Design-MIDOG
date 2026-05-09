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
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================
train_path= "./images_split/train/224_patches/"
val_path= "./images_split/val/224_patches/"
test_path= "./images_split/test/224_patches/"
train_metadata= os.path.join(train_path, 'patch_metadata.json')
val_metadata= os.path.join(val_path, 'patch_metadata.json')
test_metadata= os.path.join(test_path, 'patch_metadata.json')
# The CSV files were saved directly in the root of the repo
train_csv= "./train.csv"
val_csv= "./val.csv"
test_csv= "./test.csv"


num_epochs = 10
batch_size = 32
learning_rate = 1e-4

# LR schedule
LR_STEP_SIZE = 4             
LR_GAMMA = 0.5

# MC Dropout 相关超参数
MC_DROPOUT_RATE = 0.3
MC_NUM_FORWARD_PASSES = 20
UNCERTAINTY_THRESHOLD = 0.15


# ============================================================
# Dataset (original RGB + augmentation for train)
# ============================================================
class MitosisDataset(Dataset):
    def __init__(self, metadata_path, patches_dir, split: str = 'train'):
        self.split = split
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.patches_dir = patches_dir

        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(90)], p=0.8),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.cat_to_label = {1: 1, 2: 0}

        mitotic = sum(1 for item in self.metadata if item['category_id'] == 1)
        non_mitotic = sum(1 for item in self.metadata if item['category_id'] == 2)
        print(f'[{split}] Loaded {len(self.metadata)} patches ({mitotic} mitotic, {non_mitotic} non-mitotic)')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.patches_dir, item['patch_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.cat_to_label[item['category_id']]
        return image, label


# ============================================================
# Model: ResNet50 + MC Dropout
# ============================================================
class SimpleCNN_MCDropout(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=MC_DROPOUT_RATE):
        super(SimpleCNN_MCDropout, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# MC Dropout inference
# ============================================================
def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, images, n_forward=MC_NUM_FORWARD_PASSES):
    enable_mc_dropout(model)
    softmax = nn.Softmax(dim=-1)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_forward):
            logits = model(images)
            probs = softmax(logits)
            all_probs.append(probs.unsqueeze(0))

    all_probs = torch.cat(all_probs, dim=0)
    mean_probs = all_probs.mean(dim=0)

    eps = 1e-10
    predictive_entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=-1)
    expected_entropy = -(all_probs * torch.log(all_probs + eps)).sum(dim=-1).mean(dim=0)
    mutual_info = predictive_entropy - expected_entropy

    return mean_probs, predictive_entropy, mutual_info


# ============================================================
# Training & Evaluation
# ============================================================
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), 100. * (correct / total)


def evaluate(model, loader, loss_fn, device, desc='Evaluating'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * (correct / total)
    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy, all_preds, all_labels


def evaluate_with_uncertainty(model, loader, device, n_forward=MC_NUM_FORWARD_PASSES,
                              uncertainty_threshold=UNCERTAINTY_THRESHOLD):
    all_preds = []
    all_labels = []
    all_entropy = []
    all_mutual_info = []
    all_max_prob = []

    for images, labels in tqdm(loader, desc='MC Dropout Inference'):
        images = images.to(device)

        mean_probs, pred_entropy, mutual_info = mc_dropout_predict(
            model, images, n_forward=n_forward)

        _, predicted = mean_probs.max(1)
        max_prob, _ = mean_probs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_entropy.extend(pred_entropy.cpu().numpy())
        all_mutual_info.extend(mutual_info.cpu().numpy())
        all_max_prob.extend(max_prob.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_entropy = np.array(all_entropy)
    all_mutual_info = np.array(all_mutual_info)
    all_max_prob = np.array(all_max_prob)

    reliable_mask = all_entropy < uncertainty_threshold
    ambiguous_mask = ~reliable_mask

    n_reliable = reliable_mask.sum()
    n_ambiguous = ambiguous_mask.sum()
    n_total = len(all_labels)

    print(f'\n===== Uncertainty Analysis =====')
    print(f'  Total samples         : {n_total}')
    print(f'  Reliable (entropy < {uncertainty_threshold:.2f}): {n_reliable}  ({100*n_reliable/n_total:.1f}%)')
    print(f'  Ambiguous (entropy >= {uncertainty_threshold:.2f}): {n_ambiguous}  ({100*n_ambiguous/n_total:.1f}%)')
    print(f'  Mean predictive entropy: {all_entropy.mean():.4f}')
    print(f'  Mean mutual information: {all_mutual_info.mean():.4f}')
    print(f'  Mean max probability   : {all_max_prob.mean():.4f}')

    print(f'\n--- All Samples ---')
    print(classification_report(all_labels, all_preds,
                                target_names=['Non-mitotic', 'Mitotic'], zero_division=0))

    if n_reliable > 0:
        print(f'\n--- Reliable Subset ({n_reliable} samples) ---')
        print(classification_report(all_labels[reliable_mask], all_preds[reliable_mask],
                                    target_names=['Non-mitotic', 'Mitotic'], zero_division=0))

    if n_ambiguous > 0:
        print(f'\n--- Ambiguous Subset ({n_ambiguous} samples, recommend pathologist review) ---')
        print(classification_report(all_labels[ambiguous_mask], all_preds[ambiguous_mask],
                                    target_names=['Non-mitotic', 'Mitotic'], zero_division=0))

    return all_preds, all_labels, all_entropy, all_mutual_info, all_max_prob


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n Training history plot saved as 'training_history.png'")


def plot_uncertainty_distribution(all_entropy, all_labels, all_preds,
                                  threshold=UNCERTAINTY_THRESHOLD):
    correct_mask = (all_preds == all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.hist(all_entropy[all_labels == 0], bins=40, alpha=0.6, label='Non-mitotic', color='steelblue')
    ax.hist(all_entropy[all_labels == 1], bins=40, alpha=0.6, label='Mitotic', color='salmon')
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    ax.set_xlabel('Predictive Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty by True Label')
    ax.legend()

    ax = axes[1]
    ax.hist(all_entropy[correct_mask], bins=40, alpha=0.6, label='Correct', color='green')
    ax.hist(all_entropy[~correct_mask], bins=40, alpha=0.6, label='Incorrect', color='red')
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    ax.set_xlabel('Predictive Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty by Prediction Correctness')
    ax.legend()

    plt.tight_layout()
    plt.savefig('uncertainty_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n Uncertainty distribution plot saved as 'uncertainty_distribution.png'")


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

    print('\n Initializing model (ResNet50 + MC Dropout)...')
    model = SimpleCNN_MCDropout(num_classes=2, dropout_rate=MC_DROPOUT_RATE).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print(f'\n Starting training for {num_epochs} epochs (lr={learning_rate})...')
    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device, desc='Evaluating Val')

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'\n Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f' Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f' Current LR: {current_lr:.6g}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f' Saved best model (val_acc: {val_acc:.2f}%)')

    print('\n Plotting training history...')
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

    # ================================================================
    # Final Test: MC Dropout 
    # ================================================================
    print("\n\n========== Final Test with MC Dropout Uncertainty ==========")
    model.load_state_dict(torch.load('best_model.pth'))

    all_preds, all_labels, all_entropy, all_mutual_info, all_max_prob = \
        evaluate_with_uncertainty(model, test_loader, device,
                                 n_forward=MC_NUM_FORWARD_PASSES,
                                 uncertainty_threshold=UNCERTAINTY_THRESHOLD)

    cm = confusion_matrix(all_labels, all_preds)
    print('\nConfusion Matrix (Test Set - All):')
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f'\n Key Metrics (Test Set):')
    print(f'     Sensitivity (Recall): {sensitivity:.3f}')
    print(f'     Specificity: {specificity:.3f}')
    print(f'     Precision: {precision:.3f}')
    print(f'     F1-Score: {f1:.3f}')
    print(f'     Best Val Accuracy achieved: {best_val_acc:.2f}%')

    print('\n Plotting uncertainty distribution...')
    plot_uncertainty_distribution(all_entropy, all_labels, all_preds,
                                  threshold=UNCERTAINTY_THRESHOLD)

    results = []
    with open(test_metadata, 'r') as f:
        test_meta = json.load(f)
    for i in range(len(all_labels)):
        entry = {
            'patch_name': test_meta[i]['patch_name'] if i < len(test_meta) else f'sample_{i}',
            'true_label': int(all_labels[i]),
            'predicted_label': int(all_preds[i]),
            'max_probability': float(all_max_prob[i]),
            'predictive_entropy': float(all_entropy[i]),
            'mutual_information': float(all_mutual_info[i]),
            'is_reliable': bool(all_entropy[i] < UNCERTAINTY_THRESHOLD),
        }
        results.append(entry)

    with open('test_predictions_with_confidence.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n Per-sample results saved to 'test_predictions_with_confidence.json'")


if __name__ == "__main__":
    main()
