import os
import json
import pandas as pd


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import tensorflow as tf
from keras.models import load_model
import numpy as np
import random

random.seed(2020)
np.random.seed(2020)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"



def train(cfg):
    
    save_dir = cfg['save_dir']
    shape = (int(cfg['height']), int(cfg['width']), 3)

    n_class = int(cfg['class_number'])
    batch = int(cfg['batch'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # if cfg['model'] == 'large':
    #     from model.mobilenet_v3_large import MobileNetV3_Large
    #     model = MobileNetV3_Large(shape, n_class).build()
    # if cfg['model'] == 'small':
    #     from model.mobilenet_v3_small import MobileNetV3_Small
    #     model = MobileNetV3_Small(shape, n_class).build()

    # if cfg['model'] == 'mymodel':
    #     from model.my_model import MyModel
    #     model = MyModel(shape, n_class).build()

    # if cfg['model'] == 'v2':
    #     from model.mobilenet_v2 import MyModel
    #     model = MyModel(shape, n_class).buildRaw()

    model_path = "save/v2"
    loaded_model = load_model(os.path.join(model_path,'e_06_0.20_1.00.h5'))
    from keras import backend as K
    import tensorflow as tf
    print(loaded_model.input.op.name)
    print(loaded_model.output.op.name)
    saver = tf.train.Saver()
    saver.save(K.get_session(), 'save/ckpt/keras_model.ckpt')

    """

    python freeze_graph.py --input_meta_graph=./ckpt/keras_model.ckpt.meta --input_checkpoint=./ckpt/keras_model.ckpt --output_graph=./ckpt/keras_model.pb --output_node_names="dense/Softmax" --input_binary=true
    """

if __name__ == '__main__':
    # with open('config/config.json', 'r') as f:
    #     cfg = json.load(f)

    cfg = {
    "model": "v2",
    "height": 224,
    "width": 224,
    "class_number": 2,
    "batch": 16,
    "epochs": 50,
    "train_dir": "/home/AlgorithmicGroup/yw/workshop/antiface/data/test_position/level1/train",
    "eval_dir": "/home/AlgorithmicGroup/yw/workshop/antiface/data/test_position/level1/val",
    "save_dir": "save",
    "weights": ""
    }
    train(cfg)
    #nohup python -u train_cls.py > nohup.log 2>&1 &
