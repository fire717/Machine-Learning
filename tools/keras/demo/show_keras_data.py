import cv2
#import matplotlib.pyplot as plt
#from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
#import glob

# 设置生成器参数
datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        channel_shift_range=20)



ptrain = "/home/AlgorithmicGroup/yw/workshop/antiface/data/v3/val"
SAVE_PATH = "images/gen/"

gen_data = datagen.flow_from_directory(
        ptrain,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=True,
        save_to_dir=SAVE_PATH,
        save_prefix='gen')

# 生成9张图
for i in range(100):
    gen_data.next()

