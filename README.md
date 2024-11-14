# CDI_Data_Augmentation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Abstract:

The domain change between training and real-world datasets causes deep neural networks to frequently fail in real-world deployment despite their remarkable success in many difficult classification tasks. To overcome these constraints, we suggest Cross-Domain Interlacing (CDI), a data augmentation technique that learns robust and domain-invariant features across several domains. Current region-based dropout methods generally follow three approaches: blending content and style from regions within the same dataset, cutting and mixing patches from the same dataset, or removing pixels from training images. In contrast to these region-based dropout techniques, CDI interlaces randomly chosen areas from images of various domains and resolutions with random sections of training images. The novelty of this work lies in using augmentation sources that differ from the training data, coming from varied tasks and featuring different sizes and resolutions. CDI can be incorporated into all computer vision domains and combined with other regularization techniques. Additionally, it is easy to implement and parameter-free. We evaluated CDI across various tasks, including classification, person re-identification, and object detection, and assessed its resilience to adversarial attacks, out-of-distribution (OOD) data, and cross-domain validation. CDI achieves comparable or superior performance in classification tasks and significantly outperforms others in OOD and adversarial robustness.
![figure12](https://github.com/user-attachments/assets/0f56155d-52c0-4266-a198-cbc52b0c18a2)
