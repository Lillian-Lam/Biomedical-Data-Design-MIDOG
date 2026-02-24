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


#very basic ResNet50 backbone classifer
#CHANGE THE FILE PATH!!!!!!!!!!!
train_path = './train_224_patch/'
test_path = './test_224_patch/'
train_metadata = os.path.join(train_path, 'patch_metadata.json')
test_metadata = os.path.join(test_path, 'patch_metadata.json')
num_epochs = 20
batch_size = 32
learning_rate = 0.001

#open the ms coco file after you 224_patch_around_bbox on both the traiing and test set
#I save them as train_224_patch and test_224_patch
class MitosisDataset(Dataset):
    def __init__(self, metadata_path, patches_dir):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.patches_dir = patches_dir
        #no augmentation to the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #just basic normalization needed for ResNet50
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])])
        #mitotic(1) -> 1, non-mitotic(2) -> 0
        self.cat_to_label = {1: 1, 2: 0}
        
        mitotic = sum(1 for item in self.metadata if item['category_id'] == 1)
        non_mitotic = sum(1 for item in self.metadata if item['category_id'] == 2)
        print(f'Loaded {len(self.metadata)} patches ({mitotic} mitotic, {non_mitotic} non-mitotic)')

    def __len__(self):
        return len(self.metadata)
        
    #get the image and category id from the MSCoco file
    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.patches_dir, item['patch_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.cat_to_label[item['category_id']]
        return image, label

#basic ResNet50 backbone
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        #pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# to run each epoch
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
    
    #plot the losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    #plot accuracy 
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
    train_dataset = MitosisDataset(train_metadata, train_path)
    test_dataset = MitosisDataset(test_metadata, test_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #initialize model
    print('\n Initializing model...')
    model = SimpleCNN(num_classes=2).to(device)
    
    #loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #store the results
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    #start the training 
    print(f'\n Starting training for {num_epochs} epochs...')
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        #train
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        
        #evaluate
        test_loss, test_acc, _, _ = evaluate(model, test_loader, loss_fn, device)
        
        #store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'\n Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f' Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        #save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f' Saved best model (test_acc: {test_acc:.2f}%)')
    
    #plot the loss and accuracy 
    print('\n Plotting training history...')
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    
    print(" Final evaluation on the test set")
    #load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    _, _, all_preds, all_labels = evaluate(model, test_loader, loss_fn, device)
    
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, 
                               target_names=['Non-mitotic', 'Mitotic']))
    
    cm = confusion_matrix(all_labels, all_preds)
    print('\nConfusion Matrix:')
    print(cm)
    
    #metrics for the F1 score
    tn, fp, fn, tp = cm.ravel()
    sensitivity= tp/(tp+fn)
    specificity= tn/(tn+fp)
    precision = tp/(tp+fp)
    f1= 2 * (precision*sensitivity) /(precision+sensitivity)
    
    print(f'\n Key Metrics:')
    print(f'     Sensitivity (Recall): {sensitivity:.3f}')
    print(f'     Specificity: {specificity:.3f}')
    print(f'     Precision: {precision:.3f}')
    print(f'     F1-Score: {f1:.3f}')
    print(f'     Best Test Accuracy: {best_test_acc:.2f}%')

if __name__ == "__main__":
    main()