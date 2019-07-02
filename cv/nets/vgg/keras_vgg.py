'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Dropout, Activation, Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape
from keras.regularizers import l2
import keras.backend as K


def vgg(image_size, n_classes):
    l2_reg=0.0005
    img_height, img_width, img_channels = image_size
    x = Input(shape=(img_height, img_width, img_channels))

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv1_1')(x)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),trainable=True, name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool5')(conv5_3)

    flat = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', trainable=True, name='fc6')(flat)
    #dp6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu', trainable=True, name='fc7')(fc6)
    #dp7 = Dropout(0.5)(fc7)
    fc8 = Dense(n_classes, activation='softmax', name='fc8')(fc7)

    model = Model(inputs=x, outputs=fc8)
    return model

if __name__ == '__main__':
    image_size = (224,224,1)
    n_classes = 10
    net = vgg(image_size, n_classes)
    net.summary()