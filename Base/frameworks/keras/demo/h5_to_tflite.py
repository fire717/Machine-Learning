#coding:utf-8
#python rename.py "xx路径"  
import cv2
import os,sys
import numpy as np
from PIL import Image
import random

from keras import backend as K
import tensorflow as tf



from keras.models import load_model,save_model
my_model = load_model('model_all.h5', compile=False)
#my_model.summary()


my_model.save('model_tmp.h5')

# keras_file = './tmp/keras_model.ckpt'
# saver = tf.train.Saver()
# saver.save(K.get_session(), keras_file)
#python freeze_graph.py --input_meta_graph=./tmp/keras_model.ckpt.meta --input_checkpoint=./tmp/keras_model.ckpt --output_graph=./tmp/keras_model.pb --output_node_names="activation_6/Sigmoid" --input_binary=false
converter =  tf.lite.TocoConverter.from_keras_model_file("model_tmp.h5")
#converter.post_training_quantize = True
tflite_quantized_model=converter.convert()
 
open("model.tflite", "wb").write(tflite_quantized_model)
