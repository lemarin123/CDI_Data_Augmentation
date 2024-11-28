import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import cv2
from models.models_for_cub import ResNet
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load image list
        images_file = os.path.join(root_dir, 'images.txt')
        self.images = []
        with open(images_file, 'r') as f:
            for line in f:
                self.images.append(line.strip().split()[1])
                
        # Load class labels
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        self.labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                self.labels[img_id] = int(label) - 1
        
        # Load bounding boxes
        bbox_file = os.path.join(root_dir, 'bounding_boxes.txt')
        self.bboxes = {}
        with open(bbox_file, 'r') as f:
            for line in f:
                img_id, x, y, w, h = map(float, line.strip().split())
                self.bboxes[str(int(img_id))] = [x, y, x+w, y+h ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.images[idx])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        if self.transform:
            image = self.transform(image)
            
        img_id = str(idx + 1)
        label = self.labels[img_id]
        bbox = self.bboxes[img_id]
        
        # Scale bbox to 448x448
        scale_w = 448.0 / original_size[0]
        scale_h = 448.0 / original_size[1]
        bbox = [int(bbox[0] * scale_w), int(bbox[1] * scale_h), 
               int(bbox[2] * scale_w), int(bbox[3] * scale_h)]
        
        return image, label, bbox, img_path

def compute_iou(box_a, box_b):
    """
    Compute IoU between two bounding boxes
    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]
    Returns:
        iou: float, intersection over union
    """
    # Convert inputs to numpy arrays and ensure they're flat
    box_a = np.array(box_a).flatten()
    box_b = np.array(box_b).flatten()
    
    # Ensure boxes have correct shape
    if box_a.shape[0] != 4 or box_b.shape[0] != 4:
        raise ValueError(f"Boxes must have 4 coordinates. Got {box_a.shape[0]} and {box_b.shape[0]}")
    
    # Compute intersection
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    
    # Compute union
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)
    
    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def get_bbox_from_heatmap(heatmap, threshold=0.5):
    """
    Extract bounding box from heatmap
    Returns: [x1, y1, x2, y2]
    """
    # Normalize heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    # Threshold the heatmap
    binary = (heatmap > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.array([0, 0, 448, 448])  # Return full image if no contours found
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return np.array([x, y, x + w, y + h])

def visualize_sample(original_image, heatmap, gt_box, pred_box, save_path):
    plt.figure(figsize=(20, 5))
    
    # Original image with GT box
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.gca().add_patch(plt.Rectangle((gt_box[0], gt_box[1]), 
                                    gt_box[2] - gt_box[0],
                                    gt_box[3] - gt_box[1],
                                    fill=False, color='red', linewidth=2))
    plt.title('Ground Truth Box')
    plt.axis('off')
    
    # Original image with predicted box
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.gca().add_patch(plt.Rectangle((pred_box[0], pred_box[1]), 
                                    pred_box[2] - pred_box[0],
                                    pred_box[3] - pred_box[1],
                                    fill=False, color='green', linewidth=2))
    plt.title('Predicted Box')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 4, 3)
    plt.imshow(heatmap, cmap='jet')
    plt.title('GradCAM Heatmap')
    plt.axis('off')
    
    # Combined visualization
    plt.subplot(1, 4, 4)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.7, heatmap_colored, 0.3, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.gca().add_patch(plt.Rectangle((gt_box[0], gt_box[1]), 
                                    gt_box[2] - gt_box[0],
                                    gt_box[3] - gt_box[1],
                                    fill=False, color='red', linewidth=2))
    plt.gca().add_patch(plt.Rectangle((pred_box[0], pred_box[1]), 
                                    pred_box[2] - pred_box[0],
                                    pred_box[3] - pred_box[1],
                                    fill=False, color='green', linewidth=2))
    plt.title('Combined Visualization')
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        
        # Get the feature maps from the base_model
        for name, module in self.model.base_model._modules.items():
            if name == 'fc':  # Skip the final FC layer
                continue
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
                
        return outputs, x

class ModelOutputs():
    def __init__(self, model, target_layers, use_cuda):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)
        self.cuda = use_cuda

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = self.model.base_model.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.model.base_model.fc(output)
        return target_activations, output

class GradCAM:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        
        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)
        
    def __call__(self, input, target_class=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_class] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (448, 448))  # Resize to match the model's input size
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cub-path', type=str, required=True, help='Path to CUB_200_2011 dataset')
    parser.add_argument('--output-dir', type=str, default='gradcam_output', help='Output directory for visualization')
    parser.add_argument('--model-path', type=str, help='Path to saved model weights')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--num-visualizations', type=int, default=50, help='Number of samples to visualize')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for localization')
    parser.add_argument('--cam-threshold', type=float, default=0.5, help='Threshold for CAM heatmap')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    
    # Initialize model and GradCAM
    model = ResNet(pre_trained=True, n_class=200, model_choice=50)
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    grad_cam = GradCAM(model=model, target_layer_names=["layer4"], use_cuda=args.use_cuda)

    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

  
    dataset = CUB200Dataset(root_dir=args.cub_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    results = {
        'total_images': 0,
        'correct_localizations': 0,
        'iou_scores': [],
        'class_wise_accuracy': {}
    }

    # Process images
    for i, (inputs, labels, bbox, img_paths) in enumerate(tqdm(dataloader)):
        # Generate GradCAM
        cam = grad_cam(inputs, target_class=labels.item())
        
        # Get original image
        original_image = cv2.imread(img_paths[0])
        original_image = cv2.resize(original_image, (448, 448))
        
        # Get predicted box from heatmap
        pred_box = get_bbox_from_heatmap(cam, args.cam_threshold)
        # print(f"Predicted box shape: {pred_box.shape}")
        # print(f"Predicted box: {pred_box}")

        
        
        # Calculate IoU
        #gt_bbox = bbox[0].numpy()
        
        gt_bbox = np.array(bbox)
        gt_bbox=gt_bbox.flatten()

# Alternatively
        gt_bbox = gt_bbox.reshape(-1)
        # print(f"Ground truth box shape: {gt_bbox.shape}")
        # print(f"Ground truth box: {gt_bbox}")
        # Calculate IoU
        #gt_bbox = gt_bbox.squeeze().numpy()  # This will convert from [1,4] to [4]
        # Calculate IoU
            # Debug prints
        # print(f"Predicted box shape: {pred_box.shape}")
        # print(f"Ground truth box shape: {gt_bbox.shape}")
        # print(f"Predicted box: {pred_box}")
        # print(f"Ground truth box: {gt_bbox}")
        iou = compute_iou(pred_box, gt_bbox)
        #print(f"Iou 50: {iou}")
        #iou = compute_iou(pred_box, bbox[0].numpy())
        results['iou_scores'].append(iou)
        
        # Update metrics
        if iou >= args.iou_threshold:
            results['correct_localizations'] += 1
        results['total_images'] += 1
        
        # Save visualization for selected samples
        if i>37*50 and  i < args.num_visualizations+37*50:
            vis_path = os.path.join(vis_dir, f'sample_{i}_iou_{iou:.3f}.png')
            visualize_sample(original_image, cam, bbox, pred_box, vis_path)

    # Calculate final metrics
    loc_accuracy = results['correct_localizations'] / results['total_images']
    mean_iou = np.mean(results['iou_scores'])
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Total Images: {results['total_images']}\n")
        f.write(f"Correct Localizations: {results['correct_localizations']}\n")
        f.write(f"Localization Accuracy (IoU@{args.iou_threshold}): {loc_accuracy:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
    
    print(f"\nEvaluation Results:")
    print(f"Localization Accuracy (IoU@{args.iou_threshold}): {loc_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Results saved to {results_file}")
    print(f"Visualizations saved to {vis_dir}")

if __name__ == '__main__':
    main()
