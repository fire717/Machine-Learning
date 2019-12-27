# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017


@author: Fire
"""

# import argparse
# from keras.models import Sequential
# from model import Model
# from path import MODEL_PATH

import argparse
import cv2
import os
import numpy as np
import keras
from sklearn.utils import shuffle
from random import randint, random
from keras import backend as K

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D, Input
from keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Add, Concatenate
from keras.models import Model as MD
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from model import Model


from keras.utils.data_utils import get_file
# from keras.applications.densenet import DenseNet169
# from keras.applications.densenet import DenseNet121
# from keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.nasnet import NASNetLarge
# from keras.applications.nasnet import NASNetMobile
# from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19

import random
import os
from math import *
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf





def xception(weights=None, include_top=False, input_shape=(224, 224, 3)):
    xcpt = Xception(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = 'xception_weights_tf_dim_ordering_tf_kernels.h5'
    else:
        WRN_WEIGHTS_PATH = 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model_dir = MODEL_PATH
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    xcpt.load_weights(fpath)
    print('Xception weights loaded!')
    return xcpt





def mymodel_xception(num_classes=47):
    res = xception()
    # for layer in res.layers[:-3]:
    #     layer.trainable = False
    outputs = res.output#res.layers[-3].output
    #outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.5)(outputs)
    outputs = GlobalAveragePooling2D()(outputs)
    # outputs = Dense(128, activation='relu')(outputs)
    # outputs = Dropout(0.4)(outputs)
    outputs = Dense(num_classes, activation='softmax')(outputs)
    #outputs = Dropout(0.5)(outputs)
    model = MD(inputs=res.input, outputs=outputs)
    return model


model = mymodel_xception()


x_train, y_train, x_val, y_val = dataset.get_all_processor_data()


def train_data_generate(batch_size):
    while 1:
        iters = len(x_train)//batch_size
        for i in range(iters):
            x = []
            y = []
            for j in range(batch_size):
                imgAug = DataAugment(x_train[i*batch_size+j])
                x_data = imgAug.imgOutput()
                x_data = np.multiply(x_data, 1.0 / 127.5) - 1
                x.append(x_data)
                
                y_data = y_train[i*batch_size+j]
                y.append(y_data)

            yield (np.array(x), np.array(y))


def val_data_generate(batch_size):
    while 1:
        iters = len(x_val)//batch_size
        for i in range(iters):
            x = []
            y = []
            for j in range(batch_size):
                imgAug = DataAugment(x_val[i*batch_size+j])
                x_data = imgAug.imgOutput()
                x_data = np.multiply(x_val[i*batch_size+j], 1.0 / 127.5) - 1
                x.append(x_data)
                
                y_data = y_val[i*batch_size+j]
                y.append(y_data)

            yield (np.array(x), np.array(y))
            


early_stopping = EarlyStopping(monitor='val_loss', patience=7)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_PATH,"model.h5"),
 monitor='val_loss', save_best_only=True, save_weights_only=False)

adam = keras.optimizers.Adam(lr=1e-4) #Adam(lr=1e-4)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



batch_size = 32
epochs = 60
model.fit_generator(train_data_generate(batch_size),
            steps_per_epoch=len(x_train)//batch_size + 1, 
            epochs=epochs,
            validation_data=val_data_generate(batch_size),
            validation_steps = len(x_val)//batch_size + 1,
            verbose=1,#max_q_size=10,
            callbacks=[checkpoint, early_stopping, reduce_lr])
