import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Function
import umap

# [MOD] Hematoxylin extraction via H&E color deconvolution (HistomicsTK)
try:
    from histomicstk.preprocessing.color_deconvolution import color_deconvolution, stain_color_map
except ImportError as e:
    raise ImportError(
        "Failed to import HistomicsTK color deconvolution submodule. "
        "If you installed histomicstk with --no-deps, also install minimal deps "
        "(numpy/scipy/pillow/scikit-image). Original error: " + str(e)
    )


# ============================================================
# Config
# ============================================================
train_path = './images_split/train/224_patches'
val_path = './images_split/val/224_patches'
test_path = './images_split/test/224_patches'

train_metadata = os.path.join(train_path, 'patch_metadata.json')
val_metadata = os.path.join(val_path, 'patch_metadata.json')
test_metadata = os.path.join(test_path, 'patch_metadata.json')

train_csv = './train.csv'
val_csv = './val.csv'
test_csv = './test.csv'

# hyperparameters
num_epochs = 40          
batch_size = 32
lr_backbone = 5e-6        
lr_heads = 5e-5          
lambda_max = 1.5          
domain_loss_weight = 0.5  
mitotic_weight = 2.0
non_mitotic_weight = 1.0

domain_attr = ['Tumor', 'Species', 'Origin', 'Scanner']


# ============================================================
# Gradient reversal layer
# ============================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(torch.tensor(lambda_val))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_val,) = ctx.saved_tensors
        return -lambda_val.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, val):
        self.lambda_val = val


# ============================================================
# Dataset
# ============================================================
class MitosisDataset(Dataset):
    def __init__(self, metadata_path, patches_dir, csv_path, split='train'):
        self.split = split
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.patches_dir = patches_dir

        df = pd.read_csv(csv_path, sep=';')
        df.columns = df.columns.str.strip()
        self.slide_info = df.set_index('Slide')

        stains = ['hematoxylin', 'eosin', 'null']
        self.W = np.array([stain_color_map[st] for st in stains]).T
        self.normalize_h = transforms.Normalize(mean=[0.5], std=[0.5])

        if self.split == 'train':
            self.geometric_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                # [FIX] scale 上限改为 1.0，(0.7, 1.3) 中 >1.0 部分是无效的
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.33),
                ),
                transforms.ElasticTransform(alpha=50.0, sigma=5.0),
                # [FIX] 新增颜色抖动增强（对单通道 H 图像做亮度/对比度扰动）
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            ])
        else:
            self.geometric_aug = None

        self.cat_to_label = {1: 1, 2: 0}

        self.domain_maps = {}
        self.num_domain_classes = {}
        for attr in domain_attr:
            unique_vals = sorted(self.slide_info[attr].dropna().unique())
            self.domain_maps[attr] = {v: i for i, v in enumerate(unique_vals)}
            self.num_domain_classes[attr] = len(unique_vals)
            print(f'[{split}] {attr}: {self.num_domain_classes[attr]} classes -> {list(self.domain_maps[attr].keys())}')

        self.mitotic_count = sum(1 for item in self.metadata if item['category_id'] == 1)
        self.non_mitotic_count = sum(1 for item in self.metadata if item['category_id'] == 2)
        print(
            f'[{split}] Loaded {len(self.metadata)} patches '
            f'({self.mitotic_count} mitotic, {self.non_mitotic_count} non-mitotic)'
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.patches_dir, item['patch_name'])
        image = Image.open(img_path).convert('RGB')

        if self.split == 'train' and self.geometric_aug is not None:
            image = self.geometric_aug(image)

        rgb = np.array(image, dtype=np.uint8)[:, :, :3]
        im_deconv = color_deconvolution(rgb, self.W)
        h_img = im_deconv.StainsFloat[:, :, 0].astype(np.float32)

        p1, p99 = np.percentile(h_img, [1, 99])
        if p99 > p1:
            h_img = (h_img - p1) / (p99 - p1)
        else:
            h_img = h_img - p1
        h_img = np.clip(h_img, 0.0, 1.0)

        image = torch.from_numpy(h_img.astype(np.float32)).unsqueeze(0)
        image = self.normalize_h(image).float()

        label = self.cat_to_label[item['category_id']]

        slide_id = item['image_id']
        row = self.slide_info.loc[slide_id]
        domain_labels = {}
        for attr in domain_attr:
            domain_labels[attr] = self.domain_maps[attr][row[attr]]

        return image, label, domain_labels


# ============================================================
# Collate
# ============================================================
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    domain_labels = {}
    for attr in domain_attr:
        domain_labels[attr] = torch.tensor([item[2][attr] for item in batch], dtype=torch.long)
    return images, labels, domain_labels


# ============================================================
# Model
# ============================================================
class DANNModel(nn.Module):
    def __init__(self, num_classes=2, num_domain_classes=None, lambda_val=0.0):
        super(DANNModel, self).__init__()

        backbone = models.resnet50(pretrained=True)
        feature_dim = backbone.fc.in_features  # 2048

        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=(old_conv1.bias is not None),
        )
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
            if old_conv1.bias is not None:
                new_conv1.bias.copy_(old_conv1.bias)

        self.feature_extractor = nn.Sequential(
            new_conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        
        # 原结构: Linear(2048->512) -> ReLU -> Dropout(0.5) -> Linear(512->2)  -- 无 BN
        self.mitosis_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),      
            nn.ReLU(),
            nn.Dropout(0.6),          
            nn.Linear(512, 128),     
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self.grls = nn.ModuleDict({
            attr: GradientReversalLayer(lambda_val) for attr in domain_attr
        })

       
        self.domain_classifiers = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),  
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_domain_classes[attr]),
            )
            for attr in domain_attr
        })

    def forward(self, x):
        features = self.feature_extractor(x)
        mitosis_logits = self.mitosis_classifier(features)

        domain_logits = {}
        for attr in domain_attr:
            reversed_features = self.grls[attr](features)
            domain_logits[attr] = self.domain_classifiers[attr](reversed_features)
        return mitosis_logits, domain_logits

    def set_lambda(self, val):
        for attr in domain_attr:
            self.grls[attr].set_lambda(val)

    def predict_only(self, x):
        features = self.feature_extractor(x)
        return self.mitosis_classifier(features)


# ============================================================
# Utils
# ============================================================
def get_lambda(epoch, total_epochs, lambda_max=1.0):
    p = epoch / total_epochs
    return lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def train_one_epoch(model, loader, mitosis_loss_fn, domain_loss_fn, optimizer, device, lambda_val):
    model.train()
    model.set_lambda(lambda_val)

    total_loss = 0.0
    total_mitosis_loss = 0.0
    total_domain_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels, domain_labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        domain_labels = {attr: domain_labels[attr].to(device) for attr in domain_attr}

        optimizer.zero_grad()
        mitosis_logits, domain_logits = model(images)

        loss_mitosis = mitosis_loss_fn(mitosis_logits, labels)
        loss_domains = sum(
            domain_loss_fn(domain_logits[attr], domain_labels[attr])
            for attr in domain_attr
        ) / len(domain_attr)

        
        loss = loss_mitosis + domain_loss_weight * loss_domains

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5
        total_grad_norm += grad_norm

        total_loss += loss.item()
        total_mitosis_loss += loss_mitosis.item()
        total_domain_loss += loss_domains.item()
        _, predicted = mitosis_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        n_batches += 1

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Mit': f'{loss_mitosis.item():.4f}',
            'Dom': f'{loss_domains.item():.4f}',
            'lambda': f'{lambda_val:.3f}',
            'Grad': f'{grad_norm:.3f}',
            'Acc': f'{100. * correct / total:.1f}%'
        })

    avg_grad_norm = total_grad_norm / n_batches if n_batches > 0 else 0.0
    print(f'  Average gradient norm: {avg_grad_norm:.4f}')
    return (
        total_loss / n_batches,
        total_mitosis_loss / n_batches,
        total_domain_loss / n_batches,
        100. * correct / total,
    )


def evaluate(model, loader, mitosis_loss_fn, domain_loss_fn, device, desc='Evaluating'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    domain_correct = {attr: 0 for attr in domain_attr}
    domain_total = {attr: 0 for attr in domain_attr}
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels, domain_labels in tqdm(loader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)
            domain_labels = {attr: domain_labels[attr].to(device) for attr in domain_attr}

            mitosis_logits, domain_logits = model(images)

            loss_mitosis = mitosis_loss_fn(mitosis_logits, labels)
            loss_domains = sum(
                domain_loss_fn(domain_logits[attr], domain_labels[attr])
                for attr in domain_attr
            ) / len(domain_attr)

            running_loss += (loss_mitosis + domain_loss_weight * loss_domains).item()

            probs = softmax(mitosis_logits)[:, 1]
            _, predicted = mitosis_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for attr in domain_attr:
                _, domain_pred = domain_logits[attr].max(1)
                domain_correct[attr] += domain_pred.eq(domain_labels[attr]).sum().item()
                domain_total[attr] += domain_labels[attr].size(0)

    mitosis_acc = 100. * correct / total
    avg_loss = running_loss / len(loader)
    domain_accs = {
        attr: 100. * domain_correct[attr] / domain_total[attr]
        for attr in domain_attr
    }

    return avg_loss, mitosis_acc, domain_accs, all_preds, all_labels, all_probs


# ============================================================
# Plotting
# ============================================================
def plot_training_history(history, chance_levels):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(False)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Mitosis Accuracy')
    axes[1].legend()
    axes[1].grid(False)

    for i, attr in enumerate(domain_attr):
        ax = axes[i + 2]
        val_d_accs = [d[attr] for d in history['val_domain_accs']]
        ax.plot(epochs, val_d_accs, 'r-', label='Validation', linewidth=2)
        ax.axhline(
            y=chance_levels[attr],
            color='g',
            linestyle='--',
            label=f'Chance ({chance_levels[attr]:.1f}%)'
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Domain Acc (%)')
        ax.set_title(f'{attr} Classifier\n(lower = more invariant)')
        ax.legend(fontsize=8)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")


def plot_auc_curve(all_labels, all_probs, save_path='auc_curve.png'):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    auc_from_curve = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Test Set')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAUC/ROC curve saved to '{save_path}'")
    print(f'ROC-AUC: {auc_score:.4f}')

    return auc_score, fpr, tpr, auc_from_curve


def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    all_domain_labels = {attr: [] for attr in domain_attr}

    with torch.no_grad():
        for images, labels, domain_labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            feats = model.feature_extractor(images)
            feats = feats.view(feats.size(0), -1)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            for attr in domain_attr:
                all_domain_labels[attr].extend(domain_labels[attr].numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    all_domain_labels = {attr: np.array(all_domain_labels[attr]) for attr in domain_attr}

    print(f'Extracted features shape: {all_features.shape}')
    return all_features, all_labels, all_domain_labels


def plot_umap(features, labels, all_domain_labels, dataset, save_path='umap_visualization.png'):
    print('Computing UMAP embedding...')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)

    fig, axes = plt.subplots(1, 5, figsize=(35, 7))

    colors = ['royalblue', 'mediumvioletred']
    class_names = ['Non-mitotic', 'Mitotic']
    for i, class_name in enumerate(class_names):
        mask = labels == i
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], c=colors[i], label=class_name, s=30, alpha=0.7)
    axes[0].set_title('By True Class', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, frameon=False)
    axes[0].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for i, attr in enumerate(domain_attr):
        ax = axes[i + 1]
        domain_vals = all_domain_labels[attr]
        num_classes = len(dataset.domain_maps[attr])
        cmap = plt.cm.get_cmap('tab10', num_classes)
        inv_map = {v: k for k, v in dataset.domain_maps[attr].items()}
        for d in np.unique(domain_vals):
            mask = domain_vals == d
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[cmap(d)], label=inv_map[d], s=30, alpha=0.7)
        ax.set_title(f'By {attr}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, frameon=False, ncol=1)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('UMAP of Multi-Domain DANN Feature Extractor', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nUMAP visualization saved to {save_path}')


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\nLoading datasets...')
    train_dataset = MitosisDataset(train_metadata, train_path, train_csv, split='train')
    val_dataset = MitosisDataset(val_metadata, val_path, val_csv, split='val')
    test_dataset = MitosisDataset(test_metadata, test_path, test_csv, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    chance_levels = {attr: 100.0 / train_dataset.num_domain_classes[attr] for attr in domain_attr}
    print('\nDomain chance levels (domain acc should trend toward these):')
    for attr in domain_attr:
        print(f'  {attr}: {chance_levels[attr]:.1f}% ({train_dataset.num_domain_classes[attr]} classes)')

    class_weights = torch.tensor([non_mitotic_weight, mitotic_weight]).to(device)
    print(f'\nClass weights: Non-mitotic={non_mitotic_weight:.2f}, Mitotic={mitotic_weight:.2f}')

    mitosis_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    domain_loss_fn = nn.CrossEntropyLoss()

    print('\nInitializing multi-domain DANN model (Hematoxylin 1-ch input)...')
    model = DANNModel(
        num_classes=2,
        num_domain_classes=train_dataset.num_domain_classes,
        lambda_val=0.0,
    ).to(device)

    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr_backbone},
        {'params': model.mitosis_classifier.parameters(), 'lr': lr_heads},
        {'params': model.domain_classifiers.parameters(), 'lr': lr_heads},
        {'params': model.grls.parameters(), 'lr': lr_heads},
    ], weight_decay=1e-4)  

   
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    history = {
        'train_loss': [],
        'train_mitosis_loss': [],
        'train_domain_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_domain_accs': [],
        'lambda_vals': [],
        'lr_vals': [],
    }

    best_val_acc = 0.0
    patience = 8
    patience_counter = 0

    print(f'\nStarting multi-domain DANN training for {num_epochs} epochs...')
    print(f'  Backbone LR: {lr_backbone} | Heads LR: {lr_heads}')
    print(f'  Lambda max: {lambda_max} | Domain loss weight: {domain_loss_weight}')
    print(f'  Weight decay: 1e-4 | Early stopping patience: {patience}')

    for epoch in range(num_epochs):
        lam = get_lambda(epoch, num_epochs, lambda_max)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{num_epochs} | lambda = {lam:.4f} | backbone_lr = {current_lr:.2e}\n')

        train_loss, train_mitosis_loss, train_domain_loss, train_acc = train_one_epoch(
            model, train_loader, mitosis_loss_fn, domain_loss_fn, optimizer, device, lam
        )

       
        val_loss, val_acc, val_domain_accs, _, _, _ = evaluate(
            model, val_loader, mitosis_loss_fn, domain_loss_fn, device, desc='Evaluating Val'
        )

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_mitosis_loss'].append(train_mitosis_loss)
        history['train_domain_loss'].append(train_domain_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_domain_accs'].append(val_domain_accs)
        history['lambda_vals'].append(lam)
        history['lr_vals'].append(current_lr)

        print('\nSummary:')
        print(f'  Train Mitosis Acc: {train_acc:.2f}% | Val Mitosis Acc: {val_acc:.2f}%')
        print(f'  Train Loss: {train_loss:.4f} (Mitosis: {train_mitosis_loss:.4f}, Domain: {train_domain_loss:.4f})')
        print(f'  Val Loss: {val_loss:.4f}')
        print('\n  Val Domain Accuracies:')
        for attr in domain_attr:
            print(
                f'    {attr}: {val_domain_accs[attr]:.2f}% '
                f'(chance={chance_levels[attr]:.1f}%)'
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_dann_model.pth')
            print(f'  Saved best model (val_acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            print(f'  No improvement. Patience: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print(f'\n  Early stopping triggered at epoch {epoch + 1}.')
                break

    print('\nPlotting training history...')
    plot_training_history(history, chance_levels)

    print('\nFinal evaluation on the test set')
    model.load_state_dict(torch.load('best_dann_model.pth'))
    test_loss, test_acc, test_domain_accs, all_preds, all_labels, all_probs = evaluate(
        model, test_loader, mitosis_loss_fn, domain_loss_fn, device, desc='Evaluating Test'
    )

    print('\nClassification Report')
    print(classification_report(all_labels, all_preds, target_names=['Non-mitotic', 'Mitotic']))

    cm = confusion_matrix(all_labels, all_preds)
    print('\nConfusion Matrix:')
    print('                 Predicted')
    print('                 Non-mitotic  Mitotic')
    print(f'   Actual Non-mitotic: {cm[0,0]:6d}  {cm[0,1]:6d}')
    print(f'          Mitotic:     {cm[1,0]:6d}  {cm[1,1]:6d}')

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    roc_auc, _, _, _ = plot_auc_curve(all_labels, all_probs, save_path='auc_curve.png')

    print('\nKey Metrics:')
    print(f'  Sensitivity (Recall): {sensitivity:.3f}')
    print(f'  Specificity:          {specificity:.3f}')
    print(f'  Precision:            {precision:.3f}')
    print(f'  F1-Score:             {f1:.3f}')
    print(f'  ROC-AUC:              {roc_auc:.3f}')
    print(f'  Best Val Accuracy:    {best_val_acc:.2f}%')
    print(f'  Final Test Accuracy:  {test_acc:.2f}%')
    print(f'  Test Loss:            {test_loss:.4f}')
    print('\nFinal Domain Classifier Accuracies (test set):')
    for attr in domain_attr:
        print(f'  {attr}: {test_domain_accs[attr]:.2f}% (chance = {chance_levels[attr]:.1f}%)')

    print('\nGenerating UMAP visualizations...')
    features, true_labels, all_domain_labels = extract_features(model, test_loader, device)
    plot_umap(features, true_labels, all_domain_labels, test_dataset, 'umap_dann.png')

    print('\nAll visualizations complete!')


if __name__ == '__main__':
    main()