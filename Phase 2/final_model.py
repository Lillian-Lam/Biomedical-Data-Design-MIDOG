import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Function
import umap
import seaborn as sns

#multi-domain DANN mitosis classifier
#Feature Extractor: learns image representations (ResNet50 backbone)
#Mitosis Classifier: predicts mitotic vs non-mitotic (primary task)
#4 Domain Classifiers: one each for tumor, species, origin, scanner (adversarial tasks)
#Goal: feature extractor learns to fool all 4 domain classifiers simultaneously
#total loss = loss_mitosis + lambda_val * avg_domain_loss
#lambda ramps from 0 to lambda_max using DANN paper schedule so the mitosis
#classifier stabilises before adversarial pressure kicks in
#CHANGE THE FILE PATHS!!!!!!!!!!!
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

#model parameters
num_epochs= 50
batch_size= 32
#differential learning rates: backbone uses a much smaller lr so pretrained
#weights are only nudged, while the new classifier heads learn faster
lr_backbone= 1e-5
lr_heads= 1e-4
#lambda_max controls the ceiling of adversarial pressure from the GRL
#the schedule ramps from 0 to lambda_max over training
lambda_max= 2.0
#class weights to address the mitotic/non-mitotic imbalance
#upweighting mitotic boosts recall without changing the architecture
#these are the starting values and can be tuned later with BayesOpt
mitotic_weight= 2.0
non_mitotic_weight= 1.0

#the 4 domain attributes we adapt across simultaneously
domain_attr = ['Tumor', 'Species', 'Origin', 'Scanner']


#gradient reversal layer
#forward pass: identity function, activations pass through unchanged
#backward pass: gradients are multiplied by -lambda, which forces the feature
#extractor to STOP encoding domain information to fool the domain classifier
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(torch.tensor(lambda_val))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_val,)= ctx.saved_tensors
        #reverse and scale the gradient, return None for lambda
        return -lambda_val.item()*grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val= lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, val):
        self.lambda_val = val



#shot noise transform: simulates photon counting (Poisson) noise from the camera sensor
#each pixel value is treated as a mean photon count and sampled from a Poisson distribution
#scale controls the noise magnitude: smaller scale = more photons = finer grain
#applied on the PIL image (uint8 [0, 255]) before ToTensor so the pipeline stays unchanged
#this directly targets the Scanner domain signal because different scanners have different
#sensor gain settings that produce different shot noise profiles
class ShotNoise:
    def __init__(self, scale=0.05):
        self.scale= scale

    def __call__(self, img):
        img_array= np.array(img, dtype=np.float32)
        #map [0,255] -> photon count proxy, sample Poisson, map back
        #clipping keeps values in uint8 range after the stochastic sampling step
        photon_counts=img_array/255.0/self.scale
        noisy= np.random.poisson(photon_counts).astype(np.float32)*self.scale*255.0
        noisy= np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


#defocus blur transform: simulates out-of-focus lens aberrations from scanner hardware
#applies a large-kernel GaussianBlur with sigma drawn uniformly from (low, high)
#this covers the range from slight softening (sigma~1.5) to heavy defocus (sigma~4.0)
#using a different sigma range from the standard GaussianBlur makes these two
#augmentations complementary rather than redundant
class DefocusBlur:
    def __init__(self, kernel_size=9, sigma_low=1.5, sigma_high=4.0):
        self.kernel_size= kernel_size
        self.sigma_low=sigma_low
        self.sigma_high=sigma_high

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_low, self.sigma_high)
        return transforms.functional.gaussian_blur(img, self.kernel_size, sigma)


#open the ms coco file after you run 224_patch_around_bbox on both the training and test set
#I save them as train_224_patch and test_224_patch
#joins the json to the csv on image_id == Slide to get domain labels
#returns image, mitosis label, and 4 domain labels (tumor, species, origin, scanner)
class MitosisDataset(Dataset):
    def __init__(self, metadata_path, patches_dir, csv_path, is_train=True):
        with open(metadata_path, 'r') as f:
            self.metadata= json.load(f)
        self.patches_dir= patches_dir

        #load the csv and join to metadata on image_id == Slide
        #csv uses semicolon separator based on the MIDOG dataset format
        df= pd.read_csv(csv_path, sep=';')
        df.columns= df.columns.str.strip()
        self.slide_info= df.set_index('Slide')

        if is_train:
            #each patch looks different every epoch which prevents the model from
            #memorizing slide-level features from the small number of WSIs
            
            #RandomResizedCrop: zooms in and out to simulate cell size variation
            #scale=(0.7, 1.3) lets the cell appear 30% smaller or larger each time
            #ratio=(0.75, 1.33) adds mild aspect ratio distortion to simulate stretching
            #output is still 224x224 so the rest of the pipeline stays unchanged
            
            #ElasticTransform: locally deforms the image to simulate how cells stretch
            #and compress differently across fixation protocols and lab origins
            #this directly targets the Origin domain signal (AMC, FU Berlin, etc.)
            
            #ColorJitter: perturbs brightness/contrast/saturation to cover stain variation
            #this is a proxy for stain augmentation 
            #until we integrate the color deconvolution pipeline
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.7, 1.3),
                    ratio=(0.75, 1.33)),
                transforms.ElasticTransform(alpha=50.0, sigma=5.0),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05),
                #ShotNoise: simulates photon counting noise from the camera sensor
                #applied with p=0.5 so the model sees noisy and clean patches each epoch
                #directly targets the Scanner domain signal since sensor gain varies
                #across slide scanners - the model should ignore this noise pattern
                transforms.RandomApply([ShotNoise(scale=0.05)], p=0.5),
                #GaussianBlur: simulates mild motion blur and optical PSF variation
                #kernel_size=5 with sigma=(0.3, 1.5) covers slight softening to
                #moderate blur - targets scanner optics differences across acquisition sites
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5))], p=0.4),
                #DefocusBlur: simulates out-of-focus lens aberrations from scanner hardware
                #uses a larger kernel and sigma range than GaussianBlur above so the two
                #augmentations are complementary rather than redundant - defocus produces
                #a softer, more spread-out blur that is distinct from motion blur
                #p=0.3 keeps fully-sharp patches the most common case
                transforms.RandomApply([DefocusBlur(kernel_size=9, sigma_low=1.5, sigma_high=4.0)], p=0.3),
                transforms.ToTensor(),
                #imagenet normalization required for the pretrained ResNet50 backbone
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else:
            #test set gets no augmentation so evaluation is consistent and comparable
            #across runs - only normalize to match what the backbone expects
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        #mitotic(1) -> 1, non-mitotic(2) -> 0
        self.cat_to_label= {1: 1, 2: 0}

        #build a contiguous 0-indexed label map for each domain attribute
        #e.g. {'human breast cancer': 0, 'canine lung cancer': 1, ...}
        self.domain_maps= {}
        self.num_domain_classes= {}
        for attr in domain_attr:
            unique_vals= sorted(self.slide_info[attr].dropna().unique())
            self.domain_maps[attr] = {v: i for i, v in enumerate(unique_vals)}
            self.num_domain_classes[attr] = len(unique_vals)
            print(f' {attr}: {self.num_domain_classes[attr]} classes -> {list(self.domain_maps[attr].keys())}')

        #store class counts so we can compute weights and report imbalance
        self.mitotic_count= sum(1 for item in self.metadata if item['category_id'] == 1)
        self.non_mitotic_count= sum(1 for item in self.metadata if item['category_id'] == 2)
        print(f'Loaded {len(self.metadata)} patches ({self.mitotic_count} mitotic, {self.non_mitotic_count} non-mitotic)')

    def __len__(self):
        return len(self.metadata)

    #get the image, mitosis label, and all 4 domain labels from the MSCoco + csv files
    def __getitem__(self, idx):
        item= self.metadata[idx]
        img_path= os.path.join(self.patches_dir, item['patch_name'])
        image= Image.open(img_path).convert('RGB')
        image= self.transform(image)
        label=self.cat_to_label[item['category_id']]

        #look up the slide row in the csv and extract each domain attribute
        slide_id=item['image_id']
        row=self.slide_info.loc[slide_id]
        domain_labels={}
        for attr in domain_attr:
            domain_labels[attr]=self.domain_maps[attr][row[attr]]

        return image, label, domain_labels
    
#custom collate function to handle the domain_labels dict inside DataLoader
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    domain_labels = {}
    for attr in domain_attr:
        domain_labels[attr] = torch.tensor(
            [item[2][attr] for item in batch], dtype=torch.long)
    return images, labels, domain_labels


#multi-domain DANN model with multi-scale feature extraction and 6 components:
#Feature Extractor-> ResNet50 backbone split into 4 stages
#Multi-Scale Pooling -> pools layer2 (512-d), layer3 (1024-d), layer4 (2048-d)
#separately and concatenates to a 3584-d vector

#motivation: layer2/3 might encode coarse geometric features
#(spindle shape, chromosome arrangement, nuclear envelope breakdown) that are tumor-type invariant
#these are the features NETs share with all other tumor types 
#layer4 might encodes fine texture which works for 6/7 tumor types but fails for NETs because
#salt-and-pepper resting nuclei mimic mitotic chromatin
#by giving the classifier access to all three scales, it
#can learn to rely on geometry when texture is ambiguous

#Mitosis Classifier -> small MLP head on 3584-d features, trained normally
#   Tumor/Species/Origin/Scanner Classifiers -> adversarial heads via GRL
#each GRL independently forces the feature extractor to be invariant to that attribute
#DANN objective and augmentations are unchanged - only the feature dimension grows
class DANNModel(nn.Module):
    def __init__(self, num_classes=2, num_domain_classes=None, lambda_val=0.0):
        super(DANNModel, self).__init__()
        #num_domain_classes: dict mapping attr name -> number of classes
        #e.g. {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}

        #pretrained ResNet50 backbone split into named stages so we can tap
        #intermediate feature maps at layer2 and layer3 in addition to layer4
        backbone = models.resnet50(pretrained=True)

        #shared stem + layer1: spatial downsampling from 224 -> 56
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1)

        #layer2:56->28, output channels=51
        self.layer2=backbone.layer2
        #layer3:28->14, output channels=1024
        self.layer3 = backbone.layer3
        #layer4: 14->7, output channels=2048
        self.layer4 = backbone.layer4

        #global average pool each scale to a fixed-size vector independently
        #this preserves the scale-specific signal before concatenation
        #[B,512,1,1]
        self.pool2=nn.AdaptiveAvgPool2d(1)   
        #[B,1024,1,1]
        self.pool3=nn.AdaptiveAvgPool2d(1)   
        #[B,2048,1,1]
        self.pool4=nn.AdaptiveAvgPool2d(1)   

        #concatenated feature dimension: 512+1024+2048 = 3584
        #all downstream heads (mitosis classifier and domain classifiers) receive
        #this full multi-scale vector so DANN still adapts across all three scales
        feature_dim = 512+1024+2048  #3584

        #mitosis classifier head (primary task)
        self.mitosis_classifier=nn.Sequential(nn.Flatten(),
                                              nn.Linear(feature_dim, 512),
                                              nn.ReLU(),
                                              nn.Dropout(0.5),
                                              nn.Linear(512, num_classes))
            
        #one GRL and one domain classifier head per attribute
        #stored in ModuleDicts so pytorch tracks all parameters correctly
        self.grls=nn.ModuleDict({
            attr: GradientReversalLayer(lambda_val) for attr in domain_attr})

        self.domain_classifiers = nn.ModuleDict({
            attr: nn.Sequential(nn.Flatten(),
                                nn.Linear(feature_dim, 256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256, num_domain_classes[attr]))
            for attr in domain_attr})

    def forward(self, x):
        #compute backbone stages once and reuse to avoid redundant computation
        x=self.stem(x)
        x2=self.layer2(x)
        x3=self.layer3(x2)
        x4=self.layer4(x3)

        f2=self.pool2(x2).flatten(1)    
        f3=self.pool3(x3).flatten(1)   
        f4=self.pool4(x4).flatten(1)   
        
        #3584-d multi-scale feature vector: each scale passes through GRL separately
        #so the adversarial pressure targets all scales, not just the final layer
        features= torch.cat([f2, f3, f4], dim=1)

        mitosis_logits = self.mitosis_classifier(features)

        #each domain classifier gets its own copy of reversed multi-scale features
        #using separate GRLs means each adversary has its own lambda scale
        domain_logits = {}
        for attr in domain_attr:
            reversed_features = self.grls[attr](features)
            domain_logits[attr] = self.domain_classifiers[attr](reversed_features)

        return mitosis_logits, domain_logits

    def set_lambda(self, val):
        for attr in domain_attr:
            self.grls[attr].set_lambda(val)

    def predict_only(self, x):
        #inference only - just returns mitosis logits, skips domain heads
        x = self.stem(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = torch.cat([
            self.pool2(x2).flatten(1),
            self.pool3(x3).flatten(1),
            self.pool4(x4).flatten(1)], dim=1)
        return self.mitosis_classifier(features)

    def get_features(self, x):
        #extract 3584-d multi-scale features for UMAP visualization
        #replaces the old feature_extractor call in extract_features
        with torch.no_grad():
            x = self.stem(x)
            x2 = self.layer2(x)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return torch.cat([
                self.pool2(x2).flatten(1),
                self.pool3(x3).flatten(1),
                self.pool4(x4).flatten(1)], dim=1)


#ramp lambda from 0 to lambda_max using the schedule from the DANN paper
#starting at lambda=0 lets the mitosis classifier stabilise before the adversary kicks in
#the sigmoid shape means lambda grows quickly in the middle and plateaus at both ends
def get_lambda(epoch, total_epochs, lambda_max=1.0):
    p = epoch/total_epochs
    return lambda_max*(2.0/(1.0+np.exp(-10*p))-1.0)


#professor's adaptive weighting approach: keep mitosis and domain losses proportional
#using momentum-based running averages to smooth the adaptive weight across batches
#the key fix over the original broken version: adaptive_weight is computed from
#Python floats via .item(), so it is NOT part of the computation graph
#allows gradients to flow correctly through both loss_mitosis and loss_domains


#two scaling mechanisms work together:
#GRL uses lambda_val to control adversarial gradient strength (DANN schedule)
#adaptive_weight scales the domain loss term so it stays proportional to mitosis loss
#combined effect: adversarial signal = GRL(lambda_val)*adaptive_weight*loss_domains
#momentum=0.99 to smooth spikes without being too slow to respond (we can lower this number)
def train_one_epoch(model, loader, mitosis_loss_fn, domain_loss_fn, optimizer, device, lambda_val):
    model.train()
    model.set_lambda(lambda_val)

    total_loss=0.0
    total_mitosis_loss=0.0
    total_domain_loss=0.0
    correct=0
    total=0
    total_grad_norm=0.0
    n_batches=0

    #momentum-based running averages for the adaptive weight
    #initialised to 0.0 and seeded on the first batch
    running_mitosis_loss=0.0
    running_domain_loss=0.0
    momentum=0.99

    pbar=tqdm(loader, desc='Training')
    for batch_idx, (images, labels, domain_labels) in enumerate(pbar):
        images= images.to(device)
        labels= labels.to(device)
        domain_labels= {attr: domain_labels[attr].to(device) for attr in domain_attr}

        optimizer.zero_grad()
        mitosis_logits, domain_logits = model(images)

        loss_mitosis = mitosis_loss_fn(mitosis_logits, labels)

        #average domain losses across all 4 classifiers so the domain term
        #stays at the same scale as the mitosis loss no matter how many domains we have
        loss_domains = sum(
            domain_loss_fn(domain_logits[attr], domain_labels[attr])
            for attr in domain_attr
        ) / len(domain_attr)

        #seed the running averages from the first batch then apply momentum
        #using .item() pulls scalar floats out of the graph so adaptive_weight is never tensor 
        #this prevents the algebraic cancellation (from previous bug)
        if batch_idx == 0:
            running_mitosis_loss=loss_mitosis.item()
            running_domain_loss=loss_domains.item()
        else:
            running_mitosis_loss=momentum*running_mitosis_loss+(1-momentum)*loss_mitosis.item()
            running_domain_loss=momentum*running_domain_loss+(1-momentum)*loss_domains.item()

        #adaptive_weight is a pure Python float
        #gradients flow correctly through both loss_mitosis and loss_domains
        #the weight keeps both terms proportional as training progresses
        adaptive_weight = running_mitosis_loss/(running_domain_loss+1e-8) #prevent 0 division

        #final loss combines professor's adaptive scaling with the DANN lambda schedule
        #lambda_val ramps from 0 to lambda_max and controls GRL adversarial strength
        #adaptive_weight keeps the loss terms balanced on top of that
        loss=loss_mitosis+adaptive_weight*loss_domains

        loss.backward()


        #gradient clipping prevents instability from the sign reversal in the GRL
        #I chose 5 as it kept staying at 1. 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        #monitor gradient norm each batch to confirm weights are actually updating
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5
        total_grad_norm+=grad_norm

        total_loss+=loss.item()
        total_mitosis_loss+=loss_mitosis.item()
        total_domain_loss+=loss_domains.item()
        _, predicted=mitosis_logits.max(1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()
        n_batches += 1

        pbar.set_postfix({'Loss': f'{loss.item():.4f}',
                          'Mit': f'{loss_mitosis.item():.4f}',
                          'Dom': f'{loss_domains.item():.4f}',
                          'lambda': f'{lambda_val:.3f}',
                          'a': f'{adaptive_weight:.3f}',
                          'Grad': f'{grad_norm:.3f}',
                          'Acc': f'{100.*correct/total:.1f}%'})

    avg_grad_norm = total_grad_norm / n_batches if n_batches > 0 else 0.0
    print(f'  Average gradient norm: {avg_grad_norm:.4f}')
    return (total_loss/n_batches,
            total_mitosis_loss/n_batches,
            total_domain_loss/n_batches,
            100. * correct/total)


#now also returns all_probs: softmax probability of the mitotic class (index 1) for each patch in the loader used downstream
#to compute the ROC curve and AUC
#also returns all_domain_label_list: dict of per-attribute ground truth domain indices
#collected here so plot_f1_heatmap can slice predictions by tumor type without
#requiring a second pass through the loader
def evaluate(model, loader, mitosis_loss_fn, domain_loss_fn, device):
    model.eval()
    running_loss=0.0
    correct=0
    total=0
    all_preds=[]
    all_labels=[]
    all_probs=[] 
    #track correct predictions per domain attribute separately
    domain_correct= {attr: 0 for attr in domain_attr}
    domain_total= {attr: 0 for attr in domain_attr}
    #collect raw domain label indices per attribute so we can slice by tumor type
    all_domain_label_list = {attr: [] for attr in domain_attr}

    #track mitosis and domain losses separately so callers can plot them independently
    #this is what makes the test loss decomposition diagnostic possible
    running_mitosis_loss= 0.0
    running_domain_loss= 0.0

    with torch.no_grad():
        for images, labels, domain_labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            domain_labels = {attr: domain_labels[attr].to(device) for attr in domain_attr}

            mitosis_logits, domain_logits = model(images)

            loss_mitosis=mitosis_loss_fn(mitosis_logits, labels)
            loss_domains= sum(
                domain_loss_fn(domain_logits[attr], domain_labels[attr])
                for attr in domain_attr
            ) / len(domain_attr)

            #accumulate each component separately so we can return both to the caller
            running_mitosis_loss+=loss_mitosis.item()
            running_domain_loss+=loss_domains.item()
            #use lambda=1 for evaluation so the logged loss is comparable across epochs
            running_loss += (loss_mitosis + loss_domains).item()

            _, predicted = mitosis_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            #softmax column 1 is the probability of being mitotic
            #this is what sklearn's roc_curve expects as the score input
            probs = torch.softmax(mitosis_logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

            #count correct domain predictions per attribute
            for attr in domain_attr:
                _, domain_pred = domain_logits[attr].max(1)
                domain_correct[attr] += domain_pred.eq(domain_labels[attr]).sum().item()
                domain_total[attr] += domain_labels[attr].size(0)
                #store raw label indices for downstream per-tumor F1 slicing
                all_domain_label_list[attr].extend(domain_labels[attr].cpu().numpy())

    mitosis_acc = 100. *correct/total
    avg_loss = running_loss / len(loader)
    avg_mitosis_loss = running_mitosis_loss / len(loader)
    avg_domain_loss = running_domain_loss / len(loader)
    domain_accs = {attr: 100. *domain_correct[attr] / domain_total[attr]
                   for attr in domain_attr}

    return avg_loss, mitosis_acc, domain_accs, all_preds, all_labels, all_probs, all_domain_label_list, avg_mitosis_loss, avg_domain_loss


#history dict holds all epoch-level metrics so plotting has consistent access
#7 panels: mitosis loss, domain loss, mitosis accuracy, and one per domain attribute
def plot_training_history(history, chance_levels, results_dir='results'):
    fig, axes = plt.subplots(1, 7, figsize=(40, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    #panel 1: mitosis loss only
    axes[0].plot(epochs, history['train_mitosis_loss'], 'b-', label='Train Mitosis Loss', linewidth=2)
    axes[0].plot(epochs, history['val_mitosis_loss'], 'r-', label='Validation Mitosis Loss', linewidth=2)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Mitosis Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(False)

    #panel 2: domain loss only 
    axes[1].plot(epochs, history['train_domain_loss'], 'b-', label='Train Domain Loss', linewidth=2)
    axes[1].plot(epochs, history['val_domain_loss'], 'r-', label='Validation Domain Loss', linewidth=2)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Domain Loss\n(rising = GRL working)')
    axes[1].legend(fontsize=8)
    axes[1].grid(False)

    #panel 3: mitosis accuracy
    axes[2].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[2].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Mitosis Accuracy')
    axes[2].legend(fontsize=8)
    axes[2].grid(False)

    #panels 4-7: domain classifier accuracy per attribute
    #lower accuracy = more domain invariant = the GRL is working
    for i, attr in enumerate(domain_attr):
        ax = axes[i + 3]
        train_d_accs = [d[attr] for d in history['train_domain_accs']]
        val_d_accs = [d[attr] for d in history['val_domain_accs']]
        ax.plot(epochs, train_d_accs, 'b-', label='Training', linewidth=2)
        ax.plot(epochs, val_d_accs, 'r-', label='Validation', linewidth=2)
        #dashed green line shows what a random classifier would score
        #if domain acc trends toward this then adaptation is working
        ax.axhline(y=chance_levels[attr], color='g', linestyle='--',
                   label=f'Chance ({chance_levels[attr]:.1f}%)')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Domain Acc (%)')
        ax.set_title(f'{attr} Classifier\n(lower = more invariant)')
        ax.legend(fontsize=8)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    print("\n Training history plot saved as 'training_history.png'")


#ROC curve and AUC for the mitosis classifier on the test set
#all_probs: softmax P(mitotic) for every patch, all_labels: ground truth 0/1
#the dashed diagonal is the random-chance baseline (AUC=0.5)
#a well-trained DANN should push the curve into the top-left corner -
#high sensitivity without blowing up the false positive rate
#we annotate the AUC directly on the plot so it is readable at a glance
def plot_auc(all_labels, all_probs, save_path='auc_curve.png'):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(fpr, tpr, color='teal', linewidth=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    #dashed diagonal is random chance - anything above this is better than guessing
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
            label='Random chance (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('ROC Curve - Mitosis Classifier (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n AUC plot saved to {save_path}')
    print(f'   AUC: {roc_auc:.4f}')
    return roc_auc


def extract_features(model, dataloader, device):
    #extract 3584-d multi-scale features (layer2+layer3+layer4 concatenated)
    #used for UMAP visualization to see if domain invariance was achieved
    #multi-scale features give the UMAP a richer view: coarse geometry clusters
    #should mix across domains while fine texture clusters may still reflect tumor type
    model.eval()
    all_features = []
    all_labels = []
    all_domain_labels = {attr: [] for attr in domain_attr}

    with torch.no_grad():
        for images, labels, domain_labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            #get_features returns the same 3584-d vector the classifier sees
            feats = model.get_features(images)   # [B, 3584]
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            for attr in domain_attr:
                all_domain_labels[attr].extend(domain_labels[attr].numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    all_domain_labels = {attr: np.array(all_domain_labels[attr]) for attr in domain_attr}

    print(f'Extracted features shape: {all_features.shape}')
    return all_features, all_labels, all_domain_labels


#UMAP visualization: one panel per domain attribute plus one for true mitosis class
#if DANN is working, patches from different domains should be mixed together in
#feature space, while mitotic and non-mitotic should still be separable
def plot_umap(features, labels, all_domain_labels, dataset, save_path='umap_visualization.png'):
    print('Computing UMAP embedding...')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)

    fig, axes = plt.subplots(1, 5, figsize=(35, 7))

    #panel 1: colour by true mitosis class
    colors = ['royalblue', 'mediumvioletred']  
    class_names = ['Non-mitotic', 'Mitotic']
    for i, class_name in enumerate(class_names):
        mask = labels == i
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=colors[i], label=class_name, s=30, alpha=0.7)
    axes[0].set_title('By True Class', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, frameon=False)
    axes[0].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    #panels 2-5: one per domain attribute
    for i, attr in enumerate(domain_attr):
        ax=axes[i + 1]
        domain_vals=all_domain_labels[attr]
        num_classes=len(dataset.domain_maps[attr])
        cmap=plt.cm.get_cmap('tab10', num_classes)
        #reverse the map so we can display string labels in the legend
        inv_map={v: k for k, v in dataset.domain_maps[attr].items()}
        for d in np.unique(domain_vals):
            mask=domain_vals == d
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[cmap(d)], label=inv_map[d], s=30, alpha=0.7)
        ax.set_title(f'By {attr}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, frameon=False, ncol=1)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('UMAP of Multi-Domain DANN Feature Extractor', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n UMAP visualization saved to {save_path}')


#F1 heatmap: rows = [Non-mitotic, Mitotic], columns = tumor types
#for each tumor type subset we compute the per-class F1 so we can see whether
#the model generalises equally across all tumor types or is weak on specific ones
#NaN cells (shown in gray) appear when a tumor type has zero samples of that class
#in the test set - this is expected for rare tumors with few mitotic events
#a well-adapted DANN should produce a uniformly high mitotic row across all columns
def plot_f1_heatmap(all_preds, all_labels, all_domain_label_list, dataset,
                    save_path='f1_heatmap_tumor.png'):
    tumor_map= dataset.domain_maps['Tumor']
    #reverse the map so we can display human-readable tumor names as column headers
    inv_tumor_map= {v: k for k, v in tumor_map.items()}
    tumor_indices = np.array(all_domain_label_list['Tumor'])
    preds= np.array(all_preds)
    labels= np.array(all_labels)
    class_names= ['Non-mitotic', 'Mitotic']
    tumor_names= [inv_tumor_map[i] for i in sorted(inv_tumor_map.keys())]
    n_tumors= len(tumor_names)

    #fill with NaN by default so we can display gray for missing class/tumor combos
    f1_matrix= np.full((2, n_tumors), np.nan)

    for t_idx in range(n_tumors):
        mask= tumor_indices==t_idx
        if mask.sum()<2:
            #too few samples to compute a reliable F1 - leave as NaN
            continue
        subset_preds= preds[mask]
        subset_labels = labels[mask]
        for c_idx in range(2):
            #per-class F1 treating class c_idx as the positive label
            #zero_division=0 avoids warnings when a class is absent in the subset
            f1_matrix[c_idx, t_idx] = f1_score(
                subset_labels, subset_preds, pos_label=c_idx, zero_division=0)

    #mask NaN cells so seaborn renders them gray instead of interpolating a colour
    masked_matrix = np.ma.masked_invalid(f1_matrix)

    fig, ax = plt.subplots(figsize=(max(8, n_tumors * 1.4), 3.5))

    #draw the heatmap - RdYlGn gives an intuitive low(red)/high(green) colour scale
    sns.heatmap(masked_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=tumor_names, yticklabels=class_names,
                vmin=0.0, vmax=1.0, ax=ax, linewidths=0.6, linecolor='white',
                cbar_kws={'label': 'F1 Score', 'shrink': 0.8},
                mask=np.isnan(f1_matrix))

    #gray out cells with no data so they are visually distinct from low-F1 cells
    ax.set_facecolor('#d0d0d0')

    ax.set_title('F1 Score per Tumor Type (Test Set)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Tumor Type', fontsize=11)
    ax.set_ylabel('Class', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n F1 heatmap saved to {save_path}')


def main():
    #create the results folder if it doesn't exist so all outputs land in one place
    #exist_ok=True means no crash if the folder is already there from a previous run
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    print(f'\n Results will be saved to: {os.path.abspath(results_dir)}/')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\n Loading datasets...')
    #is_train=True applies augmentation to training set only
    #test set always gets clean transforms so evaluation is consistent across runs
    train_dataset = MitosisDataset(train_metadata, train_path, train_csv, is_train=True)
    val_dataset = MitosisDataset(val_metadata, val_path, val_csv, is_train=False)
    test_dataset = MitosisDataset(test_metadata, test_path, test_csv, is_train=False)

    #num_workers=0 loads data in the main process to avoid shared memory (/dev/shm) crashes
    #worker processes get killed by the OS when /dev/shm is too small on HPC clusters
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    #chance level for each domain attribute (what a random classifier would score)
    chance_levels = {attr: 100.0 / train_dataset.num_domain_classes[attr] for attr in domain_attr}
    print('\n Domain chance levels (domain acc should trend toward these):')
    for attr in domain_attr:
        print(f'   {attr}: {chance_levels[attr]:.1f}% ({train_dataset.num_domain_classes[attr]} classes)')

    #class weights to address the mitotic/non-mitotic imbalance
    #mitotic_weight=2.0 means the loss for misclassifying a mitosis is penalised 2x
    #this should boosts recall without changing the architecture
    class_weights = torch.tensor([non_mitotic_weight, mitotic_weight]).to(device)
    print(f'\n Class weights: Non-mitotic={non_mitotic_weight:.2f}, Mitotic={mitotic_weight:.2f}')

    #separate loss functions: mitosis gets class weights to handle imbalance
    #domain classifiers use unweighted CE because we want balanced domain discrimination
    mitosis_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    domain_loss_fn = nn.CrossEntropyLoss()

    #initialize multi-domain DANN model with lambda=0 so there is no adversarial
    #pressure at epoch 1 - the schedule ramps it up from here
    print('\n Initializing multi-domain DANN model...')
    model = DANNModel(
        num_classes=2,
        num_domain_classes=train_dataset.num_domain_classes,
        lambda_val=0.0).to(device)

    #differential learning rates: backbone stages get 1e-5 so we don't destroy the
    #pretrained ImageNet features, new heads get 1e-4 so they can actually learn
    #all four backbone stages share the same low lr - they are all pretrained weights
    optimizer = optim.Adam([
        {'params': model.stem.parameters(),   'lr': lr_backbone},
        {'params': model.layer2.parameters(), 'lr': lr_backbone},
        {'params': model.layer3.parameters(), 'lr': lr_backbone},
        {'params': model.layer4.parameters(), 'lr': lr_backbone},
        {'params': model.mitosis_classifier.parameters(), 'lr': lr_heads * 0.5},
        {'params': model.domain_classifiers.parameters(), 'lr': lr_heads},
        {'params': model.grls.parameters(),   'lr': lr_heads},
    ])

    #history dict stores all epoch-level metrics for plotting at the end
    history = {
        'train_loss': [],
        'train_mitosis_loss': [],
        'train_domain_loss': [],
        'train_acc': [],
        'val_loss': [],  
        'val_mitosis_loss': [], 
        'val_domain_loss': [],  
        'val_acc': [],  
        'train_domain_accs': [],
        'val_domain_accs': [],  
        'lambda_vals': []}
        
    best_val_acc = 0

    print(f'\n Starting multi-domain DANN training for {num_epochs} epochs...')
    print(f'   Backbone LR: {lr_backbone} | Heads LR: {lr_heads}')
    print(f'   Lambda max: {lambda_max} | Mitotic weight: {mitotic_weight}')

    for epoch in range(num_epochs):
        #ramp up lambda each epoch using the DANN paper schedule
        lam = get_lambda(epoch, num_epochs, lambda_max)
        print(f'\n')
        print(f'Epoch {epoch+1}/{num_epochs} | lambda = {lam:.4f}')
        print(f'\n')

        train_loss, train_mitosis_loss, train_domain_loss, train_acc = train_one_epoch(
            model, train_loader, mitosis_loss_fn, domain_loss_fn, optimizer, device, lam)

        #evaluate on both val and train so we can plot domain accs for both splits
        val_loss, val_acc, val_domain_accs, _, _, _, _, val_mitosis_loss, val_domain_loss = evaluate(
            model, val_loader, mitosis_loss_fn, domain_loss_fn, device)
        _, _, train_domain_accs, _, _, _, _, _, _ = evaluate(
            model, train_loader, mitosis_loss_fn, domain_loss_fn, device)

        #store everything in history for plotting
        history['train_loss'].append(train_loss)
        history['train_mitosis_loss'].append(train_mitosis_loss)
        history['train_domain_loss'].append(train_domain_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_mitosis_loss'].append(val_mitosis_loss)
        history['val_domain_loss'].append(val_domain_loss)
        history['val_acc'].append(val_acc)
        history['train_domain_accs'].append(train_domain_accs)
        history['val_domain_accs'].append(val_domain_accs)
        history['lambda_vals'].append(lam)

        print(f'\n Summary:')
        print(f'   Train Mitosis Acc: {train_acc:.2f}% | Val Mitosis Acc: {val_acc:.2f}%')
        print(f'   Train Loss: {train_loss:.4f} (Mitosis: {train_mitosis_loss:.4f}, Domain: {train_domain_loss:.4f})')
        print(f'   Val  Loss: {val_loss:.4f}  (Mitosis: {val_mitosis_loss:.4f}, Domain: {val_domain_loss:.4f})')
        print(f'\n   Domain Accuracies (train / val):')
        for attr in domain_attr:
            print(f'      {attr}: {train_domain_accs[attr]:.2f}% / {val_domain_accs[attr]:.2f}% (chance={chance_levels[attr]:.1f}%)')

        #save best model based on test accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_dann_model.pth'))
            print(f'   Saved best model (val_acc: {val_acc:.2f}%)')

    print('\n Plotting training history...')
    plot_training_history(history, chance_levels, results_dir)

    print('\n Final evaluation on the test set')
    #load the best checkpoint saved during training
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_dann_model.pth')))
    test_loss, test_acc, test_domain_accs, all_preds, all_labels, all_probs, all_domain_label_list, _, _ = evaluate(
        model, test_loader, mitosis_loss_fn, domain_loss_fn, device)

    print('\n')
    print('Classification Report')
    print(classification_report(all_labels, all_preds,
                                target_names=['Non-mitotic', 'Mitotic']))

    cm = confusion_matrix(all_labels, all_preds)
    print('\nConfusion Matrix:')
    print(f'                 Predicted')
    print(f'                 Non-mitotic  Mitotic')
    print(f'   Actual Non-mitotic: {cm[0,0]:6d}  {cm[0,1]:6d}')
    print(f'          Mitotic:     {cm[1,0]:6d}  {cm[1,1]:6d}')

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn) if (tp+fn)>0 else 0
    specificity = tn/(tn+fp) if (tn+fp)>0 else 0
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*(precision*sensitivity)/(precision+sensitivity) if (precision+sensitivity)>0 else 0

    print(f'\n Key Metrics:')
    print(f'   Sensitivity (Recall): {sensitivity:.3f}')
    print(f'   Specificity:          {specificity:.3f}')
    print(f'   Precision:            {precision:.3f}')
    print(f'   F1-Score:             {f1:.3f}')
    print(f'   Best Validation Accuracy:   {best_val_acc:.2f}%')
    print(f'\n Final Domain Classifier Accuracies (test set):')
    for attr in domain_attr:
        print(f'   {attr}: {test_domain_accs[attr]:.2f}% (chance = {chance_levels[attr]:.1f}%)')

    print('\n Generating AUC plot...')
    roc_auc = plot_auc(all_labels, all_probs, os.path.join(results_dir, 'auc_curve.png'))
    print(f'   AUC reported in KEY METRICS: {roc_auc:.4f}')

    print('\n Generating F1 heatmap by tumor type...')
    #uses the domain label list returned by the final evaluate call so no extra
    #forward pass is needed - we already have all the indices we need
    plot_f1_heatmap(all_preds, all_labels, all_domain_label_list, test_dataset,
                    os.path.join(results_dir, 'f1_heatmap_tumor.png'))

    print('\n Generating UMAP visualizations...')
    features, true_labels, all_domain_labels = extract_features(model, test_loader, device)
    plot_umap(features, true_labels, all_domain_labels, test_dataset,
              os.path.join(results_dir, 'umap_dann.png'))

    print('\n All visualizations complete!')


if __name__ == '__main__':
    main()
