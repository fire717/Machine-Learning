import argparse
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from random import randint, random
from keras import backend as K
# from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D, Input
from keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Add, Concatenate
from keras.models import Model 
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from model import Model
from path import MODEL_PATH
from mynet import xception, denseNet121, vgg16

import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard




def imgrot(img, degree, zoom):
    # 旋转, M参数：旋转中心，向左旋转度数，缩放因子，warpAffine:图像，旋转参数，输出图像大小
#   img = cv2.imread('logo2.jpg')
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, zoom)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def imgmove(img, x, y):
    # M:x方向平移100，y方向平移50, warAffine:图像，移动M，输出大小（宽，高）
#   img = cv2.imread('logo.JPG')
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def imgcolor(img):
    b, g, r = img[:,:,:1], img[:,:,1:2], img[:,:,2:3]
    b = b*random()
    b = b.astype('uint8')
    dst = np.concatenate([b, g, r], axis=-1)
    return dst

def imgrotm(img):
    rows, cols = img.shape[:2]
    a = randint(3, 8)
    pts1 = np.float32([[a, a], [20, a], [a, 20]])
    pts2 = np.float32([[1, 10], [20, a], [10, 20+a]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def imgdot(img, num=40):
    d = np.random.randint(0, 255, (5,5,3))
    dst = img.copy()
    for i in range(num):
        x = randint(55, 200)
        y = randint(55, 200)
        dst[x:x+5, y:y+5] = d
    return dst

def data_gen(trainX, trainY):
    m = []
    y = []
    for rect, f in zip(trainX, trainY):
        rot1 = imgrot(rect, randint(0, 45), randint(6, 15)/10.)
        rot2 = imgrot(rect, randint(45, 90), randint(6, 15)/10.)
        rot3 = imgrot(rect, randint(90, 135), randint(6, 15)/10.)
        rot4 = imgrot(rect, randint(135, 180), randint(6, 15)/10.)
        rot5 = imgrot(rect, randint(-45, 0), randint(6, 15)/10.)
        rot6 = imgrot(rect, randint(-90, -45), randint(6, 15)/10.)
        rot7 = imgrot(rect, randint(-135, -90), randint(6, 15)/10.)
        rot8 = imgrot(rect, randint(-180, -135), randint(6, 15)/10.)
        move = imgmove(rect, randint(-50, 50), randint(-50, 50))
        gau = cv2.GaussianBlur(rect, (3,3), 0)
        clr = imgcolor(rect)
        m.append(rot1)
        m.append(rot2)
        m.append(rot3)
        m.append(rot4)
        m.append(rot5)
        m.append(rot6)
        m.append(rot7)
        m.append(rot8)
        m.append(move)
        m.append(gau)
        m.append(clr)
        # m.append(imgrotm(rect))
        m.append(imgdot(rect))
        m.append(np.fliplr(rect))
        m.append(np.flipud(rect))
        for i in range(14):
            y.append(f)
        
    trainX = np.vstack((trainX, np.array(m)))
    # print('trainX:', trainX.shape)
    trainY = np.vstack((trainY, np.array(y)))
    # print('trainY:',trainY.shape)
    return trainX, trainY


