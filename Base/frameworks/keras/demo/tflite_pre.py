#coding:utf-8
from __future__ import print_function
import numpy as np
import cv2
from cv2 import dnn
import sys
 
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os



# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)


img = cv2.imread( "D:/Data/clothes_style/data/TestSet/img_0.jpg")
print("img shape: ", img.shape)
rows = img.shape[ 0]
cols = img.shape[ 1]
input_data = cv2.resize(img, ( 224, 224))
#input_data = np.array([input_data[:, :, [ 2, 1, 0]]]) # BGR2RGB


input_data = cv2.resize(input_data, (224, 224), interpolation=cv2.INTER_CUBIC)
input_data = np.array(input_data)
input_data = np.reshape(input_data, (1, 224, 224, 3))
input_data = input_data.astype('float32')
input_data = np.multiply(input_data, 1.0 / 255)
#input_data = np.multiply(input_data, 1.0 / 127.5) - 1



index = input_details[0]['index']
interpreter.set_tensor(index, input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print('output_data :',output_data)

