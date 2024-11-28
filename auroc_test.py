import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
from torchvision import datasets
from pyramidnet import *
from PIL import Image
import argparse

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
       
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        
        state_dict = checkpoint['state_dict']
        
       
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it exists
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded checkpoint")
        
       
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

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if args.depth == 110:
        model = PyramidNet('cifar100', 110, 64, 100, True)
    elif args.depth == 200:
        model = PyramidNet('cifar100', 200, 240, 100, True)
    else:
        raise ValueError(f"Unsupported depth: {args.depth}. Choose 110 or 200.")

   
    checkpoint_path = args.model_path if args.model_path else (
        "runs/PyraNet110_cifar100_cutout/model_best.pth.tar" if args.depth == 110 else 
        "runs/PyraNet200/model_best.pth.tar"
    )

    
    model = load_pretrained_model(model, checkpoint_path)
    model = model.to(device)
    model.eval()

   
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    #  <<<<<<<<<<    Known Data >>>>>>>>>>>>>>>>>
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    #  <<<<<<<<<<    UnKnown Data >>>>>>>>>>>>>>>>>
    unknown_dir = args.unknown_dir if args.unknown_dir else "LSUN/test"
    unknown_files = os.listdir(unknown_dir)

    batch_size = 32
    known_scores = []
    unknown_scores = []

    # Variables for accuracy calculation
    total_correct = 0
    total_samples = 0

    # Process known (CIFAR-100) data
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # Calculate confidence scores for AUROC
            softmax_probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(softmax_probs, dim=1)
            known_scores.extend(max_probs.cpu().numpy())

    # Calculate accuracy
    accuracy = (total_correct / total_samples) * 100

    # Process unknown data
    unknown_batches = [unknown_files[i:i + batch_size] for i in range(0, len(unknown_files), batch_size)]
    with torch.no_grad():
        for batch_files in unknown_batches:
            batch_X = []
            for file in batch_files:
                img = Image.open(os.path.join(unknown_dir, file)).convert('RGB')
                img = test_transform(img)
                batch_X.append(img)
            
            if batch_X:
                batch_X = torch.stack(batch_X).to(device)
                outputs = model(batch_X)
                softmax_probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(softmax_probs, dim=1)
                unknown_scores.extend(max_probs.cpu().numpy())

    known_scores = np.array(known_scores)
    unknown_scores = np.array(unknown_scores)

    # Create labels (1 for known/in-distribution, 0 for unknown/out-of-distribution)
    y_true = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])
    # Concatenate prediction scores
    y_scores = np.concatenate([known_scores, unknown_scores])

    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # Print all metrics
    print("\nEvaluation Metrics:")
    print(f"PyramidNet Depth: {args.depth}")
    print(f"Model Path: {checkpoint_path}")
    print(f"CIFAR-100 Test Accuracy: {accuracy:.2f}%")
    print(f"AUROC: {auroc:.4f}")

    print(f"\nDetailed Statistics:")
    print(f"Total test samples: {total_samples}")
    print(f"Correct predictions: {total_correct}")
    print(f"Known samples (CIFAR-100): {len(known_scores)}")
    print(f"Unknown samples: {len(unknown_scores)}")
    print(f"Known scores - Mean: {known_scores.mean():.4f}, Std: {known_scores.std():.4f}")
    print(f"Unknown scores - Mean: {unknown_scores.mean():.4f}, Std: {unknown_scores.std():.4f}")

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description='Testing AUROC for PyramidNet')
    parser.add_argument('--depth', type=int, choices=[110, 200], default=110,
                        help='Depth of PyramidNet (110 or 200)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to custom model checkpoint')
    parser.add_argument('--unknown_dir', type=str, default=None,
                        help='Path to directory with unknown test images')
    
    
    args = parser.parse_args()
    
   
    main(args)
