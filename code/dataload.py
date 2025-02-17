import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from config import *



torch.cuda.is_available()

# size = (256, 256)

import torch

torch.cuda.is_available()

from skimage import io

def tif_png_transform(file_path_name,maxmin=False):
    img = io.imread(file_path_name)
    
    if maxmin:
        if len(img.shape)==3:
            b=img.shape[-1]
            mins=img.reshape(-1,b).min(axis=0)
            maxs=img.reshape(-1,b).max(axis=0)
        else:
            mins=img.min()
            maxs=img.max()
            img = (img-mins) / (maxs-mins)
        img = img * 255 
    img = img.astype(np.uint8)
    return img



def tif_png_transform_array(img,maxmin=False):
    if maxmin:
        b=img.shape[-1]
        mins=img.reshape(-1,b).min(axis=0)
        maxs=img.reshape(-1,b).max(axis=0)
        
        # mins=img.min()
        img = (img-mins) / (maxs-mins)

        img = img * 255 
    
    
    img = img.astype(np.uint8)
    return img
class LoadData(Dataset):
    def __init__(self, images_path, masks_path):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.len = len(images_path)
        self.transform = A.Compose([
            # A.ToFloat(max_value=65535.0),
            A.Resize(height,width),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GaussNoise(),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            
            
            
        ])
        

    def __getitem__(self, idx):
        img = tif_png_transform(self.images_path[idx],MAXMIN)
        mask = tif_png_transform(self.masks_path[idx],MAXMIN)
        
        img,mask=np.array(img),np.array(mask)
        # img1=np.concatenate([img,img],axis=2)
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        # print(np.unique(mask))
        
        try:
            img = np.transpose(img, (2, 0, 1))
        except:
            img = np.expand_dims(img, axis=0)
            pass
        img = img/255
        mask= mask/255

        
        img = torch.tensor(img)

        mask = np.expand_dims(mask, axis=0)
        mask = torch.tensor(mask).long()

        return img, mask
    
    def __len__(self):
        return self.len
    
class datapre(Dataset):
    def __init__(self, images_path):
        super().__init__()
        self.images_path = images_path
        self.transform = A.Compose([
            A.Resize(height,width),
        ])
    def getite(self):
        img = tif_png_transform(self.images_path)
        
        img=np.array(img)
        transformed = self.transform(image=img)
        img = transformed['image']
        try:
            img = np.transpose(img, (2, 0, 1))
        except:
            img = np.expand_dims(img, axis=0)
            pass
        img = img/255.0
        img = torch.tensor(img)
        return img
class datapre_array(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.transform = A.Compose([
            A.Resize(height,width),
        ])
    def getite(self):
        # img = tif_png_transform(self.images_path)
        img=tif_png_transform_array(self.images,MAXMIN)
        img=np.array(img)
        transformed = self.transform(image=img)
        img = transformed['image']
        try:
            img = np.transpose(img, (2, 0, 1))
        except:
            img = np.expand_dims(img, axis=0)
            pass
        img = img/255.0
        img = torch.tensor(img)
        return img

import random
class AddPepperNoise(object):

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   
            img_[mask == 2] = 0  
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img
