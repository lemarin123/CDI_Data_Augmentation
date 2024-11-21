# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time
from transforms import Interlacing  
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np
from Cutout.util.cutout import Cutout 
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import pyramidnet_tinyimagenet as PYRM2
from models.utils import get_network
import warnings

warnings.filterwarnings("ignore")



warnings.filterwarnings("ignore")

# Add TinyImageNet dataset class
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.train = train
        
        if self.train:
            self.data_dir = os.path.join(root, 'train')
        else:
            self.data_dir = os.path.join(root, 'val')
            
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Load class mapping
        if self.train:
            # Process training data
            for class_idx, class_dir in enumerate(sorted(os.listdir(self.data_dir))):
                self.class_to_idx[class_dir] = class_idx
                class_path = os.path.join(self.data_dir, class_dir, 'images')
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
        else:
            # Process validation data
            val_annotations = os.path.join(root, 'val', 'val_annotations.txt')
            with open(val_annotations, 'r') as f:
                for line in f:
                    img_name, class_dir, *_ = line.strip().split('\t')
                    img_path = os.path.join(self.data_dir, 'images', img_name)
                    if os.path.exists(img_path):
                        if class_dir not in self.class_to_idx:
                            self.class_to_idx[class_dir] = len(self.class_to_idx)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img, label



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--entre_ratio', default=0, type=float,
                    help='entrelacement ratio')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100




def load_pretrained_model(model, checkpoint_path):
    """
    Load a pretrained model while handling module prefix issues
    Args:
        model: The model architecture instance
        checkpoint_path: Path to the checkpoint file
    Returns:
        model: Loaded model
    """
    try:
        # Load the checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get the state dict
        state_dict = checkpoint['state_dict']
        
        # Create new state dict without 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it exists
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # Load the modified state dict
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded checkpoint")
        
        # Print some checkpoint info
        if 'epoch' in checkpoint:
            print(f"Checkpoint was saved at epoch: {checkpoint['epoch']}")
        if 'best_err1' in checkpoint:
            print(f"Best error rate: {checkpoint['best_err1']}")
            
        return model
            
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        return None




         

    # ...

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


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    print("***********************************************************")
    print("train with entre ratio =",args.entre_ratio)  
    print("***********************************************************")

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        if args.entre_ratio>0.0:
            transform_train = transforms.Compose([
           
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Interlacing(probability=.8,entre=args.entre_ratio), 
            normalize,
            ])
        else:
            print("no Intrelacing augmentation, CutOut is integrated ")
            transform_train = transforms.Compose([
             
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            
            normalize,
            Cutout(n_holes=1, length=8)
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('data\\tiny-imagenet-200\\train')
        valdir = os.path.join('data\\tiny-imagenet-200\\val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        #numberofclass = 1000
        numberofclass = 200



    elif args.dataset == 'tiny-imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        
        if args.entre_ratio > 0.0:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(64),  # Tiny ImageNet is 64x64
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Interlacing(probability=.8, entre=args.entre_ratio),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = TinyImageNet(
            #os.path.join("data", 'tiny-imagenet-200'),
            "data\\tiny-imagenet-200",
            train=True,
            transform=transform_train
        )
        
        val_dataset = TinyImageNet(
            #os.path.join("data", 'tiny-imagenet-200'),
            "data\\tiny-imagenet-200",
            train=False,
            transform=transform_test
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers,
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers,
            pin_memory=True
        )
        
        numberofclass = 200  # Tiny ImageNet has 200 classes
        
        


    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet('imagenet', args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
        if args.dataset == 'tiny-imagenet':
            
            model = PYRM.PyramidNet('imagenet', args.depth, args.alpha, numberofclass,
                               args.bottleneck)
            #model = models.resnet18(pretrained=True)

# Modify the final fully connected layer to have 200 output units (for Tiny ImageNet)
            #model.fc = nn.Linear(model.fc.in_features, 200)


 
    #     checkpoint_path = "runs/PyraNet200_entre_07/model_best.pth.tar"
        if args.cutmix_prob==-1.:
            checkpoint_path = "runs/PyraNet110_entre_07_labelsmooth/model_best.pth.tar"
    

            model = load_pretrained_model(model, checkpoint_path)
    

    elif args.net_type == 'resnet18':
        #model = models.resnet18(pretrained=False)
        #model.fc = nn.Linear(model.fc.in_features, numberofclass)
        model = get_network('renset18')

    
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    #print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch>0:
            err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = err1 <= best_err1
            best_err1 = min(err1, best_err1)
            if is_best:
                best_err5 = err5

            print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_err1': best_err1,
                'best_err5': best_err5,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            #loss = criterion(output, target_a)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            
        else:
            # compute output
            output = model(input)

            loss = criterion(output, target)

            #loss = label_smoothing_loss(output, target, smoothing_factor=0.1)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >=0:
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

            
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step


        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.verbose == True:
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                    'LR: {LR:.6f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                    'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                    epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
         epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    #print('* Epoch: [{0}/{1}]\t Train Loss {loss.avg:.3f}'.format(
    #    epoch, args.epochs, loss=losses))

    return losses.avg


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


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     directory = "runs/%s/" % (args.expname)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filename = directory + filename
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save checkpoint and best model
    Args:
        state: dictionary containing model state, epoch, optimizer state etc.
        is_best: boolean indicating if this is the best model so far
        filename: name of the checkpoint file
    """
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save checkpoint
    checkpoint_path = os.path.join(directory, filename)
    torch.save(state, checkpoint_path)
    
    # Save last model weights
    model_weights_path = os.path.join(directory, 'last_model.pth')
    torch.save(state['state_dict'], model_weights_path)
    
    if is_best:
        # Save best checkpoint
        best_checkpoint_path = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_checkpoint_path)
        
        # Save best model weights
        best_model_path = os.path.join(directory, 'best_model.pth')
        torch.save(state['state_dict'], best_model_path)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar')or args.dataset.startswith('tiny') :
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet') :
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         wrong_k = batch_size - correct_k
#         res.append(wrong_k.mul_(100.0 / batch_size))

#     return res
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Option 1: Use reshape instead of view
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        
        # Option 2: Alternatively, ensure contiguous memory before view
        # correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
