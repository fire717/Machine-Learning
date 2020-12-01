#coding:utf-8
#python rename.py "xx路径"  
# tf 2.0

import os,sys
import numpy as np
import random
import functools

import tensorflow as tf
import tensorflow.keras as keras
#from keras import backend as K
#from keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.models import load_model,save_model

#from tensorflow.python.keras.utils import CustomObjectScope, get_custom_objects
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope, get_custom_objects
#my_model = load_model('mbv3_small_log4_fulltrain_9985.h5', compile=False)

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def _relu6(x):
    """Relu 6
    """
    return keras.backend.relu(x, max_value=6.0)


def get_relu6(**kwargs):
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    def _relu6(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """
        return keras.backend.relu(x, max_value=6.0)
        # if backend.backend() == 'tensorflow':
        #     try:
        #         # The native TF implementation has a more
        #         # memory-efficient gradient implementation
        #         return backend.tf.nn.swish(x)
        #     except AttributeError:
        #         pass

        # return x * backend.sigmoid(x)

    return _relu6

def get_hard_swish(**kwargs):
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    def _hard_swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """
        return x * keras.backend.relu(x + 3.0, max_value=6.0) / 6.0
        # if backend.backend() == 'tensorflow':
        #     try:
        #         # The native TF implementation has a more
        #         # memory-efficient gradient implementation
        #         return backend.tf.nn.swish(x)
        #     except AttributeError:
        #         pass

        # return x * backend.sigmoid(x)

    return _hard_swish



def _hard_swish(x):
    """Hard swish
    """
    return x * keras.backend.relu(x + 3.0, max_value=6.0) / 6.0

def inject_keras_modules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper
def init_keras_custom_objects():
    custom_objects = {
        '_relu6': inject_keras_modules(get_relu6)(),
        '_hard_swish': inject_keras_modules(get_hard_swish)()
    }

    get_custom_objects().update(custom_objects)







init_keras_custom_objects()
keras_model_path = 'mbv3_small_log4_fulltrain_9985.h5'
# with CustomObjectScope({'_hard_swish': _hard_swish, '_relu6': _relu6}):
#     saved_model = load_model(keras_model_path, compile=False)
save_model = tf.keras.models.load_model(keras_model_path)
export_dir='save'
tf.saved_model.save(save_model, export_dir)
new_model = tf.saved_model.load(export_dir)

IMAGE_WIDTH = 224 # example
with CustomObjectScope({'swish': inject_keras_modules(get_hard_swish)(),
                        '_relu6': inject_keras_modules(get_relu6)()}):
    concrete_func = new_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, IMAGE_WIDTH, IMAGE_WIDTH, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

concrete_func = new_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, IMAGE_WIDTH, IMAGE_WIDTH, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

MODEL_OUTPUT_PATH = "output.tflite"
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.allow_custom_ops = True
tflite_model = converter.convert()
open(MODEL_OUTPUT_PATH, "wb").write(tflite_model)




######### No customer things. #####################
#model = load_model('mbv3_small_log4_fulltrain_9985.h5', compile=False)
# with CustomObjectScope({'_hard_swish': _hard_swish, '_relu6': _relu6}):
#     my_model = load_model('mbv3_small_log4_fulltrain_9985.h5', compile=False)
# #my_model.summary()


# my_model.save('model_tmp.h5')

# # keras_file = './tmp/keras_model.ckpt'
# # saver = tf.train.Saver()
# # saver.save(K.get_session(), keras_file)
# #python freeze_graph.py --input_meta_graph=./tmp/keras_model.ckpt.meta --input_checkpoint=./tmp/keras_model.ckpt --output_graph=./tmp/keras_model.pb --output_node_names="activation_6/Sigmoid" --input_binary=false
#     #tflite_quantized_model=tf.lite.TocoConverter.from_keras_model_file("model_tmp.h5").convert()
#     #tflite_quantized_model=tf.contrib.lite.TFLiteConverter.from_keras_model_file(my_model).convert()
# tflite_quantized_model=tf.lite.TFLiteConverter.from_keras_model(my_model, custom_objects={'_hard_swish':_hard_swish}).convert()
 
# open("mbv3_small_log4_fulltrain_9985.tflite", "wb").write(tflite_quantized_model)
