from numpy.random.mtrand import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from datatools import crop_center

class ImageDataset(Dataset):
    def __init__(self, root, size,crop_size, scale,augmentation=0):

        self.augmentation = augmentation
        self.crop_size = crop_size
        self.size = size
        self.small_size = (int(size[0] / scale), int(size[1] / scale))
        self.imgs = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):   # get images to dataloader
        #img = Image.open(self.imgs[index % len(self.recovered)])


        label = Image.open(self.imgs[index % len(self.imgs)])
        label = crop_center(label, self.crop_size, self.crop_size)
        label = label.resize(size = self.size)
        img = label.resize(size = self.small_size)
        
        transform = transforms.Compose([
            transforms.ToTensor(),  
        ])

        img = transform(img)
        label = transform(label)
        img = img.float()
        label = label.float()
        
        imgs = {"img": img, "label": label}   # create imgs dictionary

        return imgs

    def __len__(self):  # if error num_sampler should be positive -> because Dataset not yet Downloaded
        return len(self.imgs)