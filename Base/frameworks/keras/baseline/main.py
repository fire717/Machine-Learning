#coding:utf-8
# @fire
import cv2
import os,sys
import numpy as np
from PIL import Image
import random


from my_data import myData
from my_model import myModel


def getAllName(file_dir): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
        for file in files:
            if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':
                L.append(os.path.join(root, file))
    return L





data_path_fake = "data/train/fake/"
data_path_true = "data/train/true/"
fake_imgs_train = getAllName(data_path_fake)
true_imgs_train = getAllName(data_path_true)

data_path_fake = "data/val/fake/"
data_path_true = "data/val/true/"
fake_imgs_val = getAllName(data_path_fake)
true_imgs_val = getAllName(data_path_true)




batch_size = 16
nb_epoch = 20
img_name_list_train_cate1 = true_imgs_train
img_name_list_train_cate2 = fake_imgs_train
img_name_list_val_cate1 = true_imgs_val
img_name_list_val_cate2 = fake_imgs_val
my_data = myData(batch_size, nb_epoch, img_name_list_train_cate1, img_name_list_train_cate2,
                     img_name_list_val_cate1, img_name_list_val_cate2)

print(my_data.total_train, my_data.total_val)


my_model = myModel()

my_model.train(my_data)
