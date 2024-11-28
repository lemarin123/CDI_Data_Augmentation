# Modified script based on original code
#original code:  https://github.com/alsdml/StyleMix/blob/main/test.py
# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader

import pyramidnet as PYRM

warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyramidNet CIFAR-100 Test with FGSM')
    parser.add_argument('--net_type', default='pyramidnet', type=str,
                        help='Network type: pyramidnet')
    parser.add_argument('--dataset', default='cifar100', type=str,
                        help='Dataset: cifar100')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        help='Mini-batch size (default: 128)')
    parser.add_argument('--depth', default=110, type=int,
                        help='Depth of the network (default: 110)')
    parser.add_argument('--alpha', default=64, type=float,
                        help='Number of new channel increases per depth (default: 64)')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='Path to dataset')
    parser.add_argument('--pretrained', type=str, 
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--fgsm', type=str2bool, default=False, 
                        help='Enable FGSM adversarial attack')
    parser.add_argument('--eps', default=1, type=int, 
                        help='Epsilon for FGSM attack (1, 2, 4)')
    
    return parser.parse_args()

def load_pretrained_model(model, checkpoint_path):
    """
    Load a pretrained model while handling module prefix issues
    """
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get the state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Create new state dict without 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it exists
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # Load the modified state dict
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded checkpoint")
        
        return model
            
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Parse arguments
    args = parse_arguments()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data normalization
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    
    # Test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Load validation dataset
    val_dataset = datasets.CIFAR100(
        root=args.data_dir, 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    # Model setup
    model = PYRM.PyramidNet(
        dataset='cifar100', 
        depth=args.depth, 
        alpha=args.alpha, 
        num_classes=100, 
        bottleneck=True
    )

    # Check if pretrained path is provided
    if not args.pretrained:
        raise ValueError("Please provide a path to a pretrained model checkpoint using --pretrained")

    # Load pretrained model
    model = load_pretrained_model(model, args.pretrained)
    model = model.to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {trainable_params}")

    # Prepare for validation
    cudnn.benchmark = True
    mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]], dtype=torch.float32).view(1,3,1,1).cuda()
    std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]], dtype=torch.float32).view(1,3,1,1).cuda()

    # Validate the model
    err1, err5, val_loss = validate(val_loader, model, args.fgsm, args.eps, mean, std)

    print('Accuracy (top-1 and 5 error):', err1, err5)

def validate(val_loader, model, fgsm, eps, mean, std):
    """Evaluate trained model"""
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # FGSM adversarial attack
        if fgsm:
            input_var = Variable(input, requires_grad=True)
            target_var = Variable(target)

            optimizer_input = torch.optim.SGD([input_var], lr=0.1)
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer_input.zero_grad()
            loss.backward()

            sign_data_grad = input_var.grad.sign()
            input = input * std + mean + eps / 255. * sign_data_grad
            input = torch.clamp(input, 0, 1)
            input = (input - mean)/std

        with torch.no_grad():
            input_var = Variable(input)
            target_var = Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    if fgsm:
        print('Attack (eps : {}) Prec@1 {top1.avg:.2f}'.format(eps, top1=top1))
        print('Attack (eps : {}) Prec@5 {top5.avg:.2f}'.format(eps, top5=top5))
    else:
        print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(
            top1=top1, top5=top5, error1=100-top1.avg, losses=losses))
    return top1.avg, top5.avg, losses.avg

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

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    # Compare predictions to targets
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Use reshape instead of view and handle dimension properly
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

if __name__ == '__main__':
    main()
