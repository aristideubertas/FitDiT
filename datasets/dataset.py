import torch
import numpy as np
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, Lambda, ToTensor, Normalize, InterpolationMode
import torch.nn.functional as F
from pathlib import Path
import random
import pandas as pd
import pickle
from torchvision.io import read_image
import json
from PIL import Image
import logging
logger = logging.getLogger(__name__)
import pickle
from diffusers.image_processor import VaeImageProcessor
from IPython import embed

class DenosingDitDataset:
    """
    Best aspect ratio: 256:448~9:16; 512:896~9:16; 576:1024=9:16
    """
    def __init__(
        self,
        image_path_list=['/data/zsz/ssh/zsz/work/FitDiT/data.json'],
        width = 768,
        height = 1024,
    ):                
        self.img_list = []
        for data_meta_path in image_path_list:
            self.img_list.extend(json.load(open(data_meta_path, "r")))
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)
        self.img_size = (height, width)
        self.transform = Compose([
            Resize(self.img_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize_image(self, img, target_size=768):
        width, height = img.size
        if width < height:
            scale = target_size / width
        else:
            scale = target_size / height
        
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return resized_img

    def __len__(self):
        return len(self.img_list)

    def get_sample(self, idx):
        img_list = self.img_list[idx]
        vton_img_path = img_list["vton_img_path"]
        garm_img_path = img_list["garm_img_path"]
        mask_path = img_list["mask_path"]
        pose_path = img_list["pose_path"]
        cloth_embeds_path = img_list["cloth_embeds_path"]
        
        vton_img = Image.open(vton_img_path)
        vton_img = self.resize_image(vton_img)
        
        garm_img = Image.open(garm_img_path)
        garm_img = self.resize_image(garm_img)
        
        pose_img = Image.open(pose_path)
        mask = Image.open(mask_path)
        mask = self.resize_image(mask)
        
        with open(cloth_embeds_path, 'rb') as f:  
            cloth_embeds = pickle.load(f).squeeze(0)
        
        vton_image = self.transform(vton_img)
        cloth_image = self.transform(garm_img)
        pose_image = self.transform(pose_img)
        
        mask = ToTensor()(mask)
        masked_vton_image = vton_image * (mask<0.5)
        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask, (vton_image.shape[1]//8, vton_image.shape[2]//8))[0]
                
        return dict(
            cloth_image=cloth_image,
            cloth_image_vit=cloth_embeds,
            vton_image=vton_image,
            masked_vton_image=masked_vton_image,
            mask_input=mask,
            pose_image=pose_image,
        )

    def __getitem__(self, idx):
        try:
            print(idx)
            return self.get_sample(idx)
        except Exception as e:
            logger.warning(f"Exception occurred parsing")



class GarmentDitDataset:
    """
    Best aspect ratio: 256:448~9:16; 512:896~9:16; 576:1024=9:16
    """
    def __init__(
        self,
        image_path_list=['/data/zsz/ssh/zsz/work/FitDiT/data.json'],
        width = 768,
        height = 1024,
    ):                
        self.img_list = []
        for data_meta_path in image_path_list:
            self.img_list.extend(json.load(open(data_meta_path, "r")))
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)
        self.img_size = (height, width)
        self.transform = Compose([
            Resize(self.img_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize_image(self, img, target_size=768):
        width, height = img.size
        if width < height:
            scale = target_size / width
        else:
            scale = target_size / height
        
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return resized_img

    def __len__(self):
        return len(self.img_list)

    def get_sample(self, idx):
        img_list = self.img_list[idx]
        garm_img_path = img_list["garm_img_path"]
        cloth_embeds_path = img_list["cloth_embeds_path"]
        
        garm_img = Image.open(garm_img_path)
        garm_img = self.resize_image(garm_img)
        
        
        with open(cloth_embeds_path, 'rb') as f:  
            cloth_embeds = pickle.load(f).squeeze(0)
        
        cloth_image = self.transform(garm_img)
        
        return dict(
            cloth_image=cloth_image,
            cloth_image_vit=cloth_embeds,
        )

    def __getitem__(self, idx):
        try:
            return self.get_sample(idx)
        except Exception as e:
            logger.warning(f"Exception occurred parsing")


def main():
    garm_dataset = GarmentDitDataset(
        image_path_list=['/data/zsz/ssh/zsz/work/FitDiT/data.json'],
        width = 768,
        height = 1024,
    )
    denoise_dataset = DenosingDitDataset(
        image_path_list=['/data/zsz/ssh/zsz/work/FitDiT/data.json'],
        width = 768,
        height = 1024,
    )
    # a = denoise_dataset.get_sample(1)
    print(len(garm_dataset))
    print(len(denoise_dataset))
    import torchvision
    while True:
        denoise_index = np.random.randint(0, len(denoise_dataset)-1)
        denoise_sample = denoise_dataset[denoise_index]
        ref_img = (denoise_sample['cloth_image'] + 1.0) / 2
        vton_img = (denoise_sample['vton_image'] + 1.0) / 2
        mask = (denoise_sample['mask_input'] + 1.0) / 2
        pose_image = (denoise_sample['pose_image'] + 1.0) / 2
        torchvision.utils.save_image(ref_img, 'ref_img.jpg')
        torchvision.utils.save_image(vton_img, 'vton_img.jpg')
        torchvision.utils.save_image(mask, 'mask.jpg')
        torchvision.utils.save_image(pose_image, 'pose_image.jpg')
    
        garm_index = np.random.randint(0, len(garm_dataset)-1)
        garm_sample = garm_dataset[garm_index]
        cloth_img = (garm_sample['cloth_image'] + 1.0) / 2
        torchvision.utils.save_image(cloth_img, 'cloth_img.jpg')
        
if __name__ == '__main__':
    main()
    
    
    
    
    