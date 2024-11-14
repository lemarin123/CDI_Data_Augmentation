


import os
import random
import math
from PIL import Image,ImageFile
import torchvision.transforms as transforms
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Interlacing(object):


    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,entre=0.5, image_dir= 'C:\SSD\SSD_resnet_pytorch-master\data\imagenette360'):#imagenet

 
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.entre=entre
        self.image_dir = image_dir
        self.image_paths = self._get_image_paths(image_dir)
        self.resize_transform = transforms.Resize((246, 128))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _get_image_paths(self, image_dir):
        image_paths = []
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                img_path = os.path.join(image_dir, filename)
                image_paths.append(img_path)
        return image_paths

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                if len(self.image_paths) > 0:
                    
                    selected_image_path = random.choice(self.image_paths)
                    
                   
                    selected_image = self._load_image(selected_image_path)
                    selected_image = self.resize_transform(selected_image)
                    selected_image = transforms.ToTensor()(selected_image)
                  

                    
                    if selected_image.size(1) >= h and selected_image.size(2) >= w:
                       
                        x2 = random.randint(0, selected_image.size(1) - h)
                        y2 = random.randint(0, selected_image.size(2) - w)
                        
                        patch = selected_image[:, x2:x2 + h, y2:y2 + w]
                        img[:, x1:x1 + h, y1:y1 + w] =(1-self.entre)*img[:, x1:x1 + h, y1:y1 + w]+ self.entre*patch
                        return img
        
        return img
    
