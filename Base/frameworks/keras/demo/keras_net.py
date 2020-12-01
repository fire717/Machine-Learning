#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-06 15:46:36
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

from path import MODEL_PATH
from keras.utils.data_utils import get_file
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19


# MODEL_PATH = os.path.join(sys.path[0], 'weights')

def xception(weights=None, include_top=False, input_shape=(224, 224, 3)):
    xcpt = Xception(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = 'https://www.flyai.com/m/v0.4|xception_weights_tf_dim_ordering_tf_kernels.h5'
    else:
        WRN_WEIGHTS_PATH = 'https://www.flyai.com/m/v0.4|xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model_dir = MODEL_PATH
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    xcpt.load_weights(fpath)
    print('Xception weights loaded!')
    return xcpt

def denseNet169(weights=None, include_top=False, input_shape=(224, 224, 3)):
    densenet = DenseNet169(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|densenet169_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    densenet.load_weights(fpath)

    print('Densenet169 weights loaded!')
    
    return densenet

def denseNet121(weights=None, include_top=False, input_shape=(224, 224, 3)):
    densenet = DenseNet121(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|densenet121_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    densenet.load_weights(fpath)

    print('Densenet121 weights loaded!')
    
    return densenet

def denseNet201(weights=None, include_top=False, input_shape=(224, 224, 3)):
    densenet = DenseNet201(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|densenet201_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    densenet.load_weights(fpath)

    print('Densenet201 weights loaded!')
    
    return densenet

def inceptionResNetV2(weights=None, include_top=False, input_shape=(224, 224, 3)):
    incpt = InceptionResNetV2(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.7|inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.7|inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    incpt.load_weights(fpath)

    print('InceptionResNetV2 weights loaded!')   
    return incpt

def inceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3)):
    incpt = InceptionV3(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.5|inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.5|inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    incpt.load_weights(fpath)

    print('InceptionV3 weights loaded!')   
    return incpt

def nasNetLarge(weights=None, include_top=False, input_shape=(224, 224, 3)):
    nasNet = NASNetLarge(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|NASNet-large.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|NASNet-large-no-top.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    nasNet.load_weights(fpath)

    print('NASNetLarge weights loaded!')   
    return nasNet

def nasNetMoble(weights=None, include_top=False, input_shape=(224, 224, 3)):
    nasNet = NASNetMobile(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|NASNet-mobile.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.8|NASNet-mobile-no-top.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    nasNet.load_weights(fpath)

    print('NASNetMobile weights loaded!')   
    return nasNet

def resNet50(weights=None, include_top=False, input_shape=(224, 224, 3)):
    resNet = ResNet50(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    resNet.load_weights(fpath)

    print('ResNet50 weights loaded!')   
    return resNet

def vgg16(weights=None, include_top=False, input_shape=(224, 224, 3)):
    vgg = VGG16(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.1|vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.1|vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    vgg.load_weights(fpath)

    print('VGG16 weights loaded!')   
    return vgg

def vgg19(weights=None, include_top=False, input_shape=(224, 224, 3)):
    vgg = VGG19(weights=weights, include_top=include_top, input_shape=input_shape)
    if include_top:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.1|vgg19_weights_tf_dim_ordering_tf_kernels.h5"
    else:
        WRN_WEIGHTS_PATH = "https://www.flyai.com/m/v0.1|vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = WRN_WEIGHTS_PATH.split('|')[-1]
    fpath = get_file(filename, WRN_WEIGHTS_PATH, cache_subdir=MODEL_PATH)
    vgg.load_weights(fpath)

    print('VGG19 weights loaded!')   
    return vgg


if __name__ == '__main__':
    # xceptionNet = xception()
    # desnet169 = denseNet169()
    # desnet121 = denseNet121()
    # desnet201 = denseNet201()
    # incptResV2 = inceptionResNetV2()
    # incptV3 = inceptionV3()
    # nasLarge = nasNetLarge()
    # nasMoble = nasNetMoble()
    # res50 = resNet50()
    # vggNet16 = vgg16()
    # vggNet19 = vgg19()
    pass
