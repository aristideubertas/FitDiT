import os
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn as nn
from PIL import Image
from src.utils_mask import get_mask_location
import numpy as np
import cv2
import pickle 
import random
import json

vton_img_folder = "/data/zsz/ssh/zsz/work/FitDiT/examples/model"
garm_img_folder = "/data/zsz/ssh/zsz/work/FitDiT/examples/model"
model_root = "/data/zsz/ssh/zsz/work/FitDiT/models"

image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("/data/zsz/ssh/zsz/work/FitDiT/models/image_encoder/clip-vit-large-patch14")
image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
vit_processing = CLIPImageProcessor()

dwprocessor = DWposeDetector(model_root=model_root, device="cuda:0")
parsing_model = Parsing(model_root=model_root, device='cuda:0')
category = ["Upper-body", "Upper-body", "Upper-body", "Upper-body", "Dresses", "Dresses", "Dresses", 
            "Dresses", "Dresses", "Upper-body", "Upper-body", "Upper-body", "Upper-body", "Upper-body", 
            "Upper-body", "Upper-body", "Dresses"]

def resize_image(img, target_size=768):
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

data_list = []
for i, item in enumerate(os.listdir(vton_img_folder)):  
    vton_img_path = os.path.join(vton_img_folder, item)  
    garm_img_path = vton_img_path.replace("/model", "/garment")
    vton_img = Image.open(vton_img_path)
    garm_img = Image.open(garm_img_path)
    vton_img_det = resize_image(vton_img)
    pose_image, keypoints, _, candidate = dwprocessor(np.array(vton_img_det)[:,:,::-1])
    candidate[candidate<0] = 0
    candidate = candidate[0]

    candidate[:, 0]*=vton_img_det.width
    candidate[:, 1]*=vton_img_det.height

    pose_image = pose_image[:,:,::-1] #rgb
    pose_image = Image.fromarray(pose_image)
    pose_image.save("/data/zsz/ssh/zsz/work/FitDiT/examples/pose/{}".format(item))
    pose_path = vton_img_path.replace("/model", "/pose")
    
    model_parse, _ = parsing_model(vton_img_det)
    mask, mask_gray = get_mask_location(category[i], model_parse, \
                                candidate, model_parse.width, model_parse.height)
    mask = mask.resize(vton_img.size).convert("L")
    mask_gray = mask_gray.resize(vton_img.size).convert("L")
    mask = np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:,:,np.newaxis]),axis=2)[:,:,3]
    mask = Image.fromarray(mask).convert("L")
    mask_path = vton_img_path.replace("/model", "/mask")
    mask.save(mask_path)
    
    cloth_image_vit = vit_processing(images=garm_img, return_tensors="pt").data['pixel_values']
    
    image_encoder_large = image_encoder_large.to("cuda:0")
    image_encoder_bigG = image_encoder_bigG.to("cuda:0")
    cloth_image_vit = cloth_image_vit.to("cuda:0")
    image_embeds_large = image_encoder_large(cloth_image_vit).image_embeds
    image_embeds_bigG = image_encoder_bigG(cloth_image_vit).image_embeds
    cloth_image_embeds = torch.cat([image_embeds_large, image_embeds_bigG], dim=1)
    cloth_embeds_path = vton_img_path.replace("/model", "/cloth_embeds")[:-4] + ".pkl"
    with open(cloth_embeds_path, 'wb') as f:  
        pickle.dump(cloth_image_embeds, f)  
    
    dict_ = {"vton_img_path": vton_img_path, "garm_img_path": garm_img_path, "mask_path": mask_path, "pose_path": pose_path, "cloth_embeds_path": cloth_embeds_path}
    data_list.append(dict_)

    
with open('./data.json', 'w') as json_file:  
    json.dump(data_list, json_file, indent=4)  

    
