import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import torchvision.models.resnet as resnet
from CDI_transforms import Interlacing 
import numpy as np
from Cutout.util.cutout import Cutout 
import torch.nn.functional as F
# from torchvision.transforms import v2
# mixup = v2.MixUp(num_classes=200)

import argparse
parser = argparse.ArgumentParser(description='Interlacing Tiny ImageNet Training')

parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--epochs', '--num epoch', default=90, type=int,
                    metavar='E', help='Number of epochs')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--cutmix_prob', '--proba of cutmix', default=0.0, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--Interlacing_ratio', '--alpha', default=0.0, type=float,
                    metavar='INTER', help='Alpha of intrelacing')
parser.add_argument('--cutout',  default=False, type=bool,
                    metavar='CUTOUT', help='include cutout')
parser.add_argument('--label_s',  default=False, type=bool,
                    metavar='Label_smooth', help='smoothing labels')
parser.add_argument('--mixup',  default=False, type=bool,
                    metavar='Mixup', help='include mixup')

parser.add_argument('--cutmix_modified',  default=False, type=bool,
                    metavar='cutmix_m', help='cutmix without label loss')

parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--fine_tune',  default=False, type=bool,
                    metavar='FT', help='From another model')

parser.add_argument('--test_only', action='store_true',
                    help='Run only testing on validation set')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to model checkpoint for testing')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
def mixup_data(x, y, alpha=1.0):
    """Performs mixup on the input data and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def test_model(model, val_loader):
    """Test the model and return top-1 and top-5 error rates"""
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_1 += predicted.eq(targets).sum().item()
            
            # Top-5 accuracy
            _, pred5 = outputs.topk(5, 1, largest=True, sorted=True)
            correct_5 += sum([1 for i in range(len(targets)) if targets[i] in pred5[i]])
    
    acc1 = 100. * correct_1 / total
    acc5 = 100. * correct_5 / total
    err1 = 100. - acc1
    err5 = 100. - acc5
    
    return err1, err5


class TinyImageNetValidation(Dataset):
    """Custom Dataset for Tiny ImageNet Validation Set"""
    def __init__(self, val_dir, val_annotations_file, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        
        # Read annotations file
        df = pd.read_csv(val_annotations_file, sep='\t', header=None,
                        names=['filename', 'class_id', 'class_name', 'x', 'y', 'w', 'h'])
        
        self.images = df['filename'].values
        # dictionary : class_id to a number
        unique_classes = sorted(df['class_id'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.labels = [self.class_to_idx[class_id] for class_id in df['class_id']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.val_dir, 'images', self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


BATCH_SIZE = 128
NUM_EPOCHS = 90
LEARNING_RATE = 0.
NUM_CLASSES = 200


def label_smoothing_loss(output, target, smoothing_factor=0.1):
    """
    Computes label smoothing cross-entropy loss.
    
    Args:
        output (torch.Tensor): Output logits from the model.
        target (torch.Tensor): Ground truth labels.
        smoothing_factor (float): Label smoothing factor, default is 0.1.
    
    Returns:
        torch.Tensor: Label smoothing cross-entropy loss.
    """
    log_prob = F.log_softmax(output, dim=-1)
    
    # Apply label smoothing
    targets = torch.zeros_like(log_prob).scatter_(1, target.unsqueeze(1), 1)
    targets = (1 - smoothing_factor) * targets + smoothing_factor / log_prob.size(-1)
    
    loss = (-targets * log_prob).sum(dim=1).mean()
    return loss


def load_data(data_dir,inter,cutout):
    """Load Tiny ImageNet dataset with correct validation handling"""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
   
    transform_list = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
                    ]

    #  Interlacing 
    if inter > 0.0:
        transform_list.append(Interlacing(probability=.8, entre=args.Interlacing_ratio))

    #  normalization
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    # cutout 
    if cutout:
        print("and cutout")
        transform_list.append(Cutout(n_holes=1, length=8))

   
    train_transform = transforms.Compose(transform_list)
            
    

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = ImageFolder(train_dir, transform=train_transform)
    
    # Load validation data with custom dataset

    val_dataset = TinyImageNetValidation(
        val_dir,
        val_annotations_file,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    # cut_w = np.int(W * cut_rat)
    # cut_h = np.int(H * cut_rat)
   
    # Use int() or np.int32() instead of deprecated np.int
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for inputs, targets in pbar:
 

        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        if args.mixup:
            # Apply mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        r = np.random.rand(1)
        if r < args.cutmix_prob:
            
            # generate mixed sample
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            if args.cutmix_modified:
                loss = criterion(outputs, target_a)
            else:
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            # compute output
            outputs = model(inputs)

            if args.label_s:
                loss = label_smoothing_loss(outputs, targets, smoothing_factor=0.1)
            else:
                loss = criterion(outputs, targets)

            


            #loss = label_smoothing_loss(output, target, smoothing_factor=0.1)

        


        #outputs = model(inputs)
        #loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return val_loss/len(val_loader), 100.*correct/total


class TinyResNet(nn.Module):
    """Modified ResNet architecture for 64x64 images"""
    def __init__(self, block, layers, num_classes=200):
        super(TinyResNet, self).__init__()
        
        # Modified initial layers for smaller input size
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Remove maxpool layer as it's too aggressive for 64x64 images
        
        # Main ResNet layers with modified strides
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # No stride in first block
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)    # Output: 64x64x64
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)   # Output: 64x64x64
        x = self.layer2(x)   # Output: 32x32x128
        x = self.layer3(x)   # Output: 16x16x256
        x = self.layer4(x)   # Output: 8x8x512
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_tiny_resnet18(num_classes=200):
    """Create ResNet18 model modified for tiny images"""
    return TinyResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes)

def create_tiny_resnet34(num_classes=200):
    """Create ResNet34 model modified for tiny images"""
    return TinyResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes)

def create_tiny_resnet50(num_classes=200):
    """Create ResNet50 model modified for tiny images"""
    return TinyResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes)



def main():
 

    global args
    args = parser.parse_args()
    LEARNING_RATE=args.lr
    NUM_EPOCHS=args.epochs
    print(LEARNING_RATE)

    model = create_tiny_resnet50(num_classes=200)
    
    
    pretrained_model = torchvision.models.resnet50(pretrained=True)
   
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    
   
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if args.fine_tune:
        checkpoint = torch.load("C:\\data_augmentation\\runs_tiny_imagnet\\Entre_08\\resnet_tiny_imagenet_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])

    train_loader, val_loader = load_data('C:\\data_augmentation\\data\\tiny-imagenet-200',args.Interlacing_ratio,args.cutout)
    model = model.to(device)
    
    if args.test_only:
        if args.model_path is None:
            raise ValueError("Must provide --model_path when using --test_only")
        
        # Load checkpoint
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test the model
        err1, err5 = test_model(model, val_loader)
        print(f'Test Results:')
        print(f'Top-1 Error: {err1:.2f}%')
        print(f'Top-5 Error: {err5:.2f}%')
        print(f'Top-1 Accuracy: {100-err1:.2f}%')
        print(f'Top-5 Accuracy: {100-err5:.2f}%')
        return
    else:
        


        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                            momentum=0.9, weight_decay=5e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        # Load data
        
        
        # Training loop
        
        directory = "runs_tiny_imagnet/%s/" % (args.expname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_weights_path = os.path.join(directory, 'resnet_tiny_imagenet_best.pth')
        
        best_acc = 0
        for epoch in range(NUM_EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion)
            
            print(f'Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                print('Saving model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': val_acc,
                },model_weights_path )
                best_acc = val_acc
            
            scheduler.step()

if __name__ == '__main__':
    main()
