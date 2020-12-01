#coding:utf-8
# from __future__ import print_function
import numpy as np
import cv2
# from cv2 import dnn
# import sys
 
# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# import os

# import time




import onnxruntime as rt

model_path = 'keras_model.onnx'
sess=rt.InferenceSession(model_path)#model_path就是模型的地址
input_name=sess.get_inputs()[0].name


img = cv2.imread( './1593250301105_1f43f6a0e8.png')
print("img shape: ", img.shape)
rows = img.shape[ 0]
cols = img.shape[ 1]
inp = cv2.resize(img, ( 224, 224))
inp = inp[:, :, [ 2, 1, 0]] # BGR2RGB
inp = inp/255.0
data = inp.reshape( 1, inp.shape[ 0], inp.shape[ 1], 3)
print(data.shape)
data = np.transpose(data,(0,3,1,2))
print(data.shape)
data = data.astype(np.float32)

res=sess.run(None,{input_name:data})

print("res: ", res)
