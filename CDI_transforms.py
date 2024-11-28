


import os
import random
import math
from PIL import Image,ImageFile
import torchvision.transforms as transforms
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Interlacing(object):


    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,entre=0.5, image_dir= 'C:\SSD\SSD_resnet_pytorch-master\data\imagenette360'):#imagenette360
        #'C:\\Users\\elbahri\\Documents\\re_ident\\imagnette\\data'):#
#"C:\\Users\\elbahri\\Documents\\re_ident\\reid-strong-baseline-master\\data\\market1501\\bounding_box_train"):                 
        #'C:\\Users\\elbahri\\Documents\\re_ident\\reid-strong-baseline-master\\data\\dukemtmc-reid\DukeMTMC-reID\\bounding_box_train'):#
 
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
                    # Randomly select an image path from the list
                    selected_image_path = random.choice(self.image_paths)
                    
                    # Load and resize the selected image
                    selected_image = self._load_image(selected_image_path)
                    selected_image = self.resize_transform(selected_image)
                    selected_image = transforms.ToTensor()(selected_image)
                    #selected_image = self.normalize(selected_image)

                    # Ensure the selected image is larger than the patch size
                    if selected_image.size(1) >= h and selected_image.size(2) >= w:
                        # Randomly select a patch from the selected image
                        x2 = random.randint(0, selected_image.size(1) - h)
                        y2 = random.randint(0, selected_image.size(2) - w)
                        
                        patch = selected_image[:, x2:x2 + h, y2:y2 + w]
                        img[:, x1:x1 + h, y1:y1 + w] =(1-self.entre)*img[:, x1:x1 + h, y1:y1 + w]+ self.entre*patch
                        return img
        
        return img
    


import os
import random
import math
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

class Entrlacing2(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, entre=0.5, 
                 image_dir='C:\SSD\SSD_resnet_pytorch-master\data\imagenette360'):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.entre = entre
        self.image_dir = image_dir
        self.image_paths = self._get_image_paths(image_dir)
        self.resize_transform = transforms.Resize((246, 128))

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

        width, height = img.size
        area = height * width

        for attempt in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, height - h)
                y1 = random.randint(0, width - w)

                if len(self.image_paths) > 0:
                    # Randomly select and load an image
                    selected_image_path = random.choice(self.image_paths)
                    selected_image = self._load_image(selected_image_path)
                    selected_image = self.resize_transform(selected_image)

                    # Ensure the selected image is larger than the patch size
                    if selected_image.size[0] >= w and selected_image.size[1] >= h:
                        # Convert images to numpy arrays for easier manipulation
                        img_array = np.array(img)
                        selected_array = np.array(selected_image)

                        # Randomly select a patch from the selected image
                        x2 = random.randint(0, selected_image.size[1] - h)
                        y2 = random.randint(0, selected_image.size[0] - w)

                        # Extract and blend patches
                        original_patch = img_array[x1:x1 + h, y1:y1 + w, :]
                        selected_patch = selected_array[x2:x2 + h, y2:y2 + w, :]
                        
                        # Blend patches
                        blended_patch = (1 - self.entre) * original_patch + self.entre * selected_patch
                        blended_patch = blended_patch.astype(np.uint8)
                        
                        # Apply the blended patch back to the original image
                        img_array[x1:x1 + h, y1:y1 + w, :] = blended_patch
                        
                        # Convert back to PIL Image
                        return Image.fromarray(img_array)

        return img