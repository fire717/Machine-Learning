#coding:utf-8
from __future__ import print_function
import numpy as np
import cv2
from cv2 import dnn
import sys
 
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os


model_name = 'frozen_insightface_r50.pb'

# #Read the graph.
# with tf.gfile.FastGFile(os.path.join(model_dir, model_name), 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
# print("11111")
# with tf.Session() as sess:
# # Restore session
#     sess.graph.as_default()
#     print("22222")
#     tf.import_graph_def(graph_def, name= '')
#     # Read and preprocess an image.
#     img = cv2.imread( '../t7.png')
#     print("img shape: ", img.shape)
#     rows = img.shape[ 0]
#     cols = img.shape[ 1]
#     inp = cv2.resize(img, ( 112, 112))
#     inp = inp[:, :, [ 2, 1, 0]] # BGR2RGB
#     # Run the model
#     out = sess.run([sess.graph.get_tensor_by_name( 'output:0'),],
#         feed_dict={ 'data:0': inp.reshape( 1, inp.shape[ 0], inp.shape[ 1], 3)})
#     # Visualize detected bounding boxes.
#     print("out: ", out)
#     # detections = int(out[ 0][ 0])
#     # print(detections)

# def get_ga(data):

#     ret = data[0]
    
#     print("ret length: ", len(ret[0]))

#     #ret = ret1

#     g = ret[:,0:2].flatten()
#     gender = np.argmax(g)
#     a = ret[:,2:202].reshape( (100,2) )
#     a = np.argmax(a, axis=1)
#     age = int(sum(a))

#     return gender, age

# gender, age = get_ga(out)
# print(gender, age)




# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)


img = cv2.imread( '../head112.jpg')
print("img shape: ", img.shape)
rows = img.shape[ 0]
cols = img.shape[ 1]
input_data = cv2.resize(img, ( 112, 112))
input_data = np.array([input_data[:, :, [ 2, 1, 0]]]) # BGR2RGB
print(input_data.shape)
input_data = input_data.astype(np.float32)
index = input_details[0]['index']
interpreter.set_tensor(index, input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print('output_data shape:',output_data.shape)

def get_ga(data):
    print("ret length: ", len(data))
    g = data[:,0:2].flatten()
    gender = np.argmax(g)
    a = data[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))
    return gender, age

print("output_data max : ", np.argmax(output_data))

print(output_data[0,0],output_data[0,1], output_data[0,163],output_data[0,164] )
print(output_data[0,164]/np.sum(output_data[0,:]))
gender, age = get_ga(output_data)
print(gender, age)
