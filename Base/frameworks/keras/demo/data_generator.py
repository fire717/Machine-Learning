# -*- coding: utf-8 -*-

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



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(img_path + ID )
            X[i,] = np.array(img)

            # Store class
            y[i] = self.labels[ID]
        #print(" in __data_generation: ", X.shape)
        #print(" in __data_generation: ", y.shape)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def mymodel(num_classes=10):
    res = denseNet121()
    # for layer in res.layers[:-3]:
    #     layer.trainable = False
    outputs = res.output
    outputs = GlobalAveragePooling2D()(outputs)
    outputs = Dense(num_classes, activation='softmax')(outputs)
    model = Model(inputs=res.input, outputs=outputs)
    return model
# model1 = xception()
# model2 = denseNet121()
# sqeue = model_merge(model1, model2, num_classes=6)
sqeue = mymodel()

'''
dataset.get_step() 获取数据的总迭代次数

'''
# sqeue.summary()

# Parameters
params = {'dim': (224,224),
          'batch_size': 4,
          'n_classes': 10,
          'n_channels': 3,
          'shuffle': True}

# Datasets
data_path = "data/img/"
img_path = data_path + "mnist/"
labels = data_path+"labels.txt"
"""
like:
1.jpg 0
2.jpg 9
3.jpg 2
...
"""

with open(labels, 'r') as f:
    lines = f.readlines()

assert len(lines) == 55000

partition = {}# IDs
labels = {}# Labels
train_ids = []
val_ids = []
total_num = 6000#len(lines)
for i in range(total_num):
    name, label = lines[i].strip().split(" ")
    if i<int(total_num*0.9):
        train_ids.append(name)
    else:
        val_ids.append(name)

    labels[name] = int(label)
partition['train'] = train_ids
partition['validation'] = val_ids

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)






# x_train, y_train = data_gen(x_train, y_train)
# x_train, y_train = shuffle(x_train, y_train)
# # x_train = x_train/255.
# # x_val = x_val/255.
# print('x_train:', x_train.shape, y_train.shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, min_lr=1e-6)
savemodel = SaveModel(sqeue)

checkpoint = ModelCheckpoint(filepath='./data/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5',
 monitor='val_loss', save_best_only=False, save_weights_only=True)

sqeue.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# history = sqeue.fit(x_train, y_train,
#                         epochs=30,
#                         batch_size=2,
#                         validation_split=0.1,
#                         verbose=2,
#                         callbacks=[checkpoint, early_stopping, reduce_lr])


history = sqeue.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                        epochs=30,
                        verbose=1,
                        callbacks=[checkpoint, early_stopping, reduce_lr])
