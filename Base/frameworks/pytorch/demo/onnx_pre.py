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


import time

import onnxruntime as rt

model_path = 'mymodel.onnx'
sess=rt.InferenceSession(model_path)#model_path就是模型的地址
input_name=sess.get_inputs()[0].name


img = cv2.imread( 'tmp/face0_0.8583003.jpg')
print("img shape: ", img.shape)
inp = cv2.resize(img, ( 112, 112))
inp = inp[:, :, [ 2, 1, 0]] # BGR2RGB

data = inp.reshape( 1, inp.shape[ 0], inp.shape[ 1], 3)
#print(data.shape)
data = np.transpose(data,(0,3,1,2))
data = data/255.0
data = (data-0.5)/0.5
#print(data.shape)
data = data.astype(np.float32)

for _ in range(5):
    t = time.time()
    res=sess.run(None,{input_name:data})[0]
    print(time.time() - t)

print("res: ", res[0][:20])
print("res: ", np.array(res).shape)
