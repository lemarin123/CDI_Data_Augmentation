# CDI_Data_Augmentation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Submitted paper to CVPR 2025
## Title: Cross-Domain Interlacing Data Augmentation: A Regularization Approach for Multi-Domain Computer Vision

## Abstract:

The domain change between training and real-world datasets causes deep neural networks to frequently fail in real-world deployment despite their remarkable success in many difficult classification tasks. To overcome these constraints, we suggest Cross-Domain Interlacing (CDI), a data augmentation technique that learns robust and domain-invariant features across several domains. Current region-based dropout methods generally follow three approaches: blending content and style from regions within the same dataset, cutting and mixing patches from the same dataset, or removing pixels from training images. In contrast to these region-based dropout techniques, CDI interlaces randomly chosen areas from images of various domains and resolutions with random sections of training images. The novelty of this work lies in using augmentation sources that differ from the training data, coming from varied tasks and featuring different sizes and resolutions. CDI can be incorporated into all computer vision domains and combined with other regularization techniques. Additionally, it is easy to implement and parameter-free. We evaluated CDI across various tasks, including classification, person re-identification, and object detection, and assessed its resilience to adversarial attacks, out-of-distribution (OOD) data, and cross-domain validation. CDI achieves comparable or superior performance in classification tasks and significantly outperforms others in OOD and adversarial robustness.
![figure12](https://github.com/user-attachments/assets/0f56155d-52c0-4266-a198-cbc52b0c18a2)

## Getting Started
### Requirements
- Python >3
- Torch 2.4.1+cu212 
- torchvision 0.19.1+cu121
- NumPy 1.26
- tqdm
### Train Examples
- CIFAR-100 CIFAR 10 on:  Pyramid-200-240, Pyramid-110-64, resnet18.
- The  testing code on cifar100 and cifar10 with pyramid net is from: [CutMix](https://github.com/clovaai/CutMix-PyTorch)
- The training code is a modified version as it supports ResNet-18 and multiple regularization techniques.
- Use arguments:
  -inter_ratio  >0 To train with CDI, example:  --inter_ratio  0.7
  -SLCDI True  To train SLCDI
  -cutmix_prob  >0 To train with Cutmix , example:  -- cutmix_prob 0.5
  -cutout True  To train with cutout
  -dataset to choose dataset exemple: --dataset cifar100 or  --dataset cifar10
  -net_type to choose the network: resnet18 or pyramidnet
for pyramidnet , select : --depth 110 --alpha 64 or  --depth 200 --alpha 240    
```
- example 1: training pyramidnet 110 with SLCDI  
python train.py --net_type pyramidnet  --depth 110 --alpha 64   --dataset cifar100  --batch_size 64 --lr 0.25 --expname PyraNet200_inter_08_test1 --epochs 300 --beta 1.0 --cutmix_prob 0.0 --no-verbose --inter_ratio 0.8 --SLCDI True --fine_tune False --cutout False

- example 2: training Resnet-18 on cifar-100 with SLCDI 
python train.py --net_type resnet18 --dataset cifar100  --batch_size 64 --lr 0.25 --expname resnet_experiment --epochs 300 --beta 1.0 --cutmix_prob 0.0 --no-verbose --inter_ratio 0.8 --SLCDI False

```
- Tiny-Imagnet:
Download Tiny-imagenet from [here](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
 
- Training:
```
python resnet-training.py --expname cutout --cutmix_prob 0.0  --lr 0.1 --cutout False --Interlacing_ratio 0.7
```
- Testing:
Replace model_path 
```
python resnet-training.py --test_only  --model_path  /set/your/model/path/resnet_tiny_imagenet_best.pth
```

### Test example 
- Test your trained models:
```
python test.py --net_type pyramidnet  --dataset cifar100  --batch_size 128 --depth 200 --alpha 240  --pretrained /set/your/model/path/model_best.pth.tar
```
## WSOL results
![sample_1875_iou_0 806](https://github.com/user-attachments/assets/3f96ddab-cf19-4655-9324-03aaa948d91b)

