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
inter_ratio  >0 To train with CDI, example:  --inter_ratio  0.7
SLCDI True  To train SLCDI
  cutmix_prob  >0 To train with Cutmix , example:  -- cutmix_prob 0.5
  cutout True  To train with cutout
  dataset to choose dataset exemple: --dataset cifar100 or  --dataset cifar10
  net_type to choose the network: resnet18 or pyramidnet
  for pyramidnet , select : --depth 110 --alpha 64 or  --depth 200 --alpha 240    
```
- example 1: training pyramidnet 110 with SLCDI  
python train.py --net_type pyramidnet  --depth 110 --alpha 64   --dataset cifar100  --batch_size 64 --lr 0.25 --expname PyraNet200_inter_08_test1 --epochs 300 --beta 1.0 --cutmix_prob 0.0 --no-verbose --inter_ratio 0.8 --SLCDI True --fine_tune False --cutout False

- example 2: training Resnet-18 on cifar-100 with CDI 
python train.py --net_type resnet18 --dataset cifar100  --batch_size 64 --lr 0.25 --expname resnet_experiment --epochs 300 --beta 1.0 --cutmix_prob 0.0 --no-verbose --inter_ratio 0.8 --SLCDI False

```
- Tiny-Imagnet:
Download Tiny-imagenet from [here](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
 
- Training:
```
python resnet-training.py --expname tiny_img_inter07 --cutmix_prob 0.0  --lr 0.1 --cutout False --Interlacing_ratio 0.7
```
- Testing:
Replace model_path 
```
python resnet-training.py --test_only  --model_path  /set/your/model/path/resnet_tiny_imagenet_best.pth
```

### Classification tests
- Test your trained models:
```
python test.py --net_type pyramidnet  --dataset cifar100  --batch_size 128 --depth 200 --alpha 240  --pretrained /set/your/model/path/model_best.pth.tar
```

### Test of  Robustness to Adversarial Attacks
- Test your trained models against Adversial attacks: We have the best score :)
```
python fgsm_test.py --net_type pyramidnet  --batch_size 128 --depth 200 --alpha 240 --fgsm True --eps 1  --pretrained  /set/your/model/path/model_best.pth.tar
```


### Test of Uncertainty
- Test your trained models:
```
python auroc_test.py --depth 200 --model_path   /set/your/model/path/model_best.pth.tar --unknown_dir /set/your/dataset/path/LSUN/test
```



<h2 id="experiments">Experimental Results and Pretrained Models</h2>

- PyramidNet-200 pretrained on CIFAR-100 dataset:

Method | Top-1 Error | FDSM | AUROC |Model file
-- | -- | -- | -- | -- 
PyramidNet-200 [[CVPR'17](https://arxiv.org/abs/1610.02915)] (baseline) | 16.45 | [model](https://www.dropbox.com/sh/6rfew3lr761jq6c/AADrdQOXNx5tWmgOSnAw9NEVa?dl=0)
PyramidNet-200 + CutMix | **14.23** |60.09|88.70| [model](https://www.dropbox.com/sh/o68qbvayptt2rz5/AACy3o779BxoRqw6_GQf_QFQa?dl=0)

PyramidNet-200 + Mixup [[ICLR'18](https://arxiv.org/abs/1710.09412)] | 15.63 |52.75|81.36| [model](https://www.dropbox.com/sh/g55jnsv62v0n59s/AAC9LPg-LjlnBn4ttKs6vr7Ka?dl=0)
PyramidNet-200 + Manifold Mixup [[ICML'19](https://arxiv.org/abs/1806.05236)] | 16.14 |-|-| [model](https://www.dropbox.com/sh/nngw7hhk1e8msbr/AABkdCsP0ABnQJDBX7LQVj4la?dl=0)
PyramidNet-200 + Cutout [[arXiv'17](https://arxiv.org/abs/1708.04552)] | 16.53 |52.75|81.36| [model](https://www.dropbox.com/sh/ajjz4q8c8t6qva9/AAAeBGb2Q4TnJMW0JAzeVSpfa?dl=0)
PyramidNet-200 + **SLCDI** | 15.92 |**46.64**|85.63| [model](https://www.dropbox.com/scl/fi/qt2wik50w5jx93gxgrtxq/SLCDI_pyramid200_model_best.pth.tar?rlkey=ispmx8gd38xhckjrb195zorku&st=pminmaux&dl=0)
PyramidNet-200 + **CDI** | 15.84 |73.55|**92.06**| [model](https://www.dropbox.com/scl/fi/fgqny8o8li0122zrgv1qc/CDI_pyramid_200_model_best.pth.tar?rlkey=sd4igem2ob0j0nmsn6j1hhl03&st=z0p7vbhk&dl=0)


## WSOL results
![sample_1875_iou_0 806](https://github.com/user-attachments/assets/3f96ddab-cf19-4655-9324-03aaa948d91b)

