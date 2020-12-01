#coding:utf-8
from __future__ import print_function
import numpy as np
import cv2
from cv2 import dnn
import sys
 
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



#### data
img = cv2.imread( '1593250301105_1f43f6a0e8.png')
print("img shape: ", img.shape)
rows = img.shape[ 0]
cols = img.shape[ 1]
img = cv2.resize(img, ( 224, 224))
#img = img[:, :, [ 2, 1, 0]] # BGR2RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.multiply(img, 1.0 / 255.0)





#### model
sess = tf.Session()
saver = tf.train.import_meta_graph('./keras_model.ckpt.meta') # 加载模型结构
saver.restore(sess, tf.train.latest_checkpoint('./')) # 只需要指定目录就可以恢复所有变量信息


# 获取placeholder变量
input_x = sess.graph.get_tensor_by_name('input_1:0')

# 获取需要进行计算的operator
op = sess.graph.get_tensor_by_name('dense_1/Softmax:0')

ret = sess.run(op, 
            feed_dict={ input_x: np.array([img],dtype = np.float32)})
print("ret: ",ret)



