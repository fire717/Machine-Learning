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

from keras.layers.normalization import BatchNormalization


def alexnet(image_size, n_classes):

    img_height, img_width, img_channels = image_size

    x = Input(shape=(img_height, img_width, img_channels))

    conv1 = Conv2D(96, (11, 11), strides=(4,4), activation='relu', padding='valid', name='conv1')(x)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
    bn1 = BatchNormalization(axis=-1)(pool1)

    conv2 = Conv2D(256, (5, 5), strides=(1,1), activation='relu', padding='same', name='conv2')(bn1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2')(conv2)
    bn2 = BatchNormalization(axis=-1)(pool2)

    conv3 = Conv2D(384, (3, 3), strides=(1,1), activation='relu', padding='same', name='conv3')(bn2)
    conv4 = Conv2D(384, (3, 3), strides=(1,1), activation='relu', padding='same', name='conv4')(conv3)
    conv5 = Conv2D(256, (3, 3), strides=(1,1), activation='relu', padding='same', name='conv5')(conv4)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(conv5)
    
    flat = Flatten()(pool3)
    fc6 = Dense(4096, activation='relu', trainable=True, name='fc6')(flat)
    dp6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu', trainable=True, name='fc7')(dp6)
    dp7 = Dropout(0.5)(fc7)
    fc8 = Dense(n_classes, activation='softmax', name='fc8')(dp7)

    model = Model(inputs=x, outputs=fc8)
    return model
