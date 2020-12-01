from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
import random
from torch.utils.data.dataset import Dataset
import cv2
import torchvision.transforms as transforms
# import imagehash
from PIL import Image
from torchsummary import summary
import torchvision.models as models
import pretrainedmodels
#from pretrainedmodels.models.xception import Xception,xception

#print(pretrainedmodels.pretrained_settings['xception'])
#{'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth', 'input_space': 'RGB', 'input_size': [3, 299, 299], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'num_classes': 1000, 'scale': 0.8975}}

#b

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)

def getAllName(file_dir, tail_list = ['.png','.jpg']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L



img_path_list = getAllName("../../mywork/data/datasets/raw/train_clean/train_pad")
transform = transforms.Compose([
                            # transforms.Resize((224, 224)),
                            # transforms.CenterCrop(size=(210, 180)),
                            transforms.Resize((224, 224)),
                            #transforms.RandomAffine(20, translate=(0.2,0.1), scale=(0.9,1.1),shear=(10,10), fillcolor=(0,0,0)),
                            #transforms.RandomHorizontalFlip(),
                            # transforms.RandomRotation(20),
                            #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.2),
                            #transforms.ToTensor(),
                             #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                             ])


for i,img_path in enumerate(img_path_list):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img.save("tmp/"+str(i)+".jpg", quality=100)

    if i>100:
        break

