import os
import json
import pandas as pd

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

import cv2
import numpy as np


model = None
model_shape = None

def getAllName(file_dir, tail_list = ['.jpg']):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L

def pre(img, save_model = False):

    global model
    global model_shape
    if model is None:
        with open('config/config.json', 'r') as f:
            cfg = json.load(f)

        save_dir = cfg['save_dir']
        model_shape = (int(cfg['height']), int(cfg['width']), 3)
        n_class = int(cfg['class_number'])
        batch = int(cfg['batch'])

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if cfg['model'] == 'large':
            from model.mobilenet_v3_large import MobileNetV3_Large
            model = MobileNetV3_Large(model_shape, n_class).build()
        if cfg['model'] == 'small':
            from model.mobilenet_v3_small import MobileNetV3_Small
            model = MobileNetV3_Small(model_shape, n_class).build()

        if cfg['model'] == 'mymodel':
            from model.my_model import MyModel
            model = MyModel(model_shape, n_class).build()

        if cfg['model'] == 'v2':
            from model.mobilenet_v2 import MyModel
            model = MyModel(model_shape, n_class).build()


        pre_weights = "save/v3_weights0.87-0.87.h5"#cfg['weights']
        if pre_weights and os.path.exists(pre_weights):
            model.load_weights(pre_weights, by_name=True)
            print("------------------load pre model!!!!!")

    if(save_model):
        print("Finish save.")
        model.save('save/model_all.h5')

    # 预处理
    img = cv2.resize(img, model_shape[:2])
    img = img*1. / 255



    pre = model.predict(np.array([img]))
    #print("pre: ",pre,np.argmax(pre[0]))

    #pre_cate = np.argmax(pre[0])
    return pre


def activaLearning(pre_probs):
    #对预测结果打分
    score = -np.sum(pre_probs * np.log(pre_probs + 1e-14), axis=1)
    return score


def getPreCate(img):
    pre_probs = pre(img)
    #print(pre_probs)
    """
    img_0  v3
[[6.1250734e-01 2.1597120e-01 2.3435018e-04 1.2239005e-02 1.5422572e-01
  2.3779788e-03 1.4025271e-04 2.1053634e-03 1.8515548e-04 1.3541079e-05]]
    """
    pre_cate = np.argmax(pre_probs[0])
    return pre_cate


def calCateAcc(dir_path):
    #统计已标注数据集每个类别的准确率
    cate_num = 10
    acc_list = []
    for i in range(cate_num):
        img_list = getAllName(os.path.join(dir_path, str(i)))
        right_count = 0
        #scores = []

        for j,img_path in enumerate(img_list):
            img = cv2.imread(img_path)
            pre_probs = pre(img)
            acl_score = activaLearning(pre_probs)
            #print("acl_score: ",acl_score)
            pre_cate = np.argmax(pre_probs[0])
            #print("pre: ",pre_probs,pre_cate)
            #print("---------------")
            if pre_cate == i:
                right_count+=1

            #scores.append(acl_score)
        #print("score: ", np.max(scores),np.min(scores),np.mean(scores))#1.8640124 0.14783148 1.0553739
        acc = right_count*1.0/len(img_list)
        acc_list.append(acc)
        print("cate "+str(i)+" acc is: "+str(acc))
        """
        v1
        cate 0 acc is: 0.4796747967479675
        cate 1 acc is: 0.45714285714285713
        cate 2 acc is: 0.7352941176470589
        cate 3 acc is: 0.5288461538461539
        cate 4 acc is: 0.4158415841584158
        cate 5 acc is: 0.46534653465346537
        cate 6 acc is: 0.9038461538461539
        cate 7 acc is: 0.6504854368932039
        cate 8 acc is: 0.5436893203883495
        cate 9 acc is: 0.7757009345794392
        59

        v2
        cate 0 acc is: 0.17073170731707318
        cate 1 acc is: 0.45714285714285713
        cate 2 acc is: 0.5490196078431373
        cate 3 acc is: 0.6057692307692307
        cate 4 acc is: 0.5346534653465347
        cate 5 acc is: 0.5445544554455446
        cate 6 acc is: 0.875
        cate 7 acc is: 0.6504854368932039
        cate 8 acc is: 0.5436893203883495
        cate 9 acc is: 0.7289719626168224
        56.1

        v3
        cate 0 acc is: 0.5526315789473685
        cate 1 acc is: 0.706766917293233
        cate 2 acc is: 0.8870967741935484
        cate 3 acc is: 0.944954128440367
        cate 4 acc is: 0.8309859154929577
        cate 5 acc is: 0.8703703703703703
        cate 6 acc is: 0.9397590361445783
        cate 7 acc is: 0.952054794520548
        cate 8 acc is: 0.8983050847457628
        cate 9 acc is: 0.896551724137931
        avg acc:  0.8479476324286666
        """
    print("avg acc: ", np.mean(acc_list))
        


def moveNoLabelImg(read_dir, save_dir, mode = 1):
    #mode0 粗筛未标注的主动学习得分高的数据,直接手动标注
    #mode1 粗筛置信度高的类别，减少标注量
    #mode2 清洗训练集 和标注类别不一样都移动
    cate_num = 10
    for i in range(cate_num):
        save_dir_cate = os.path.join(save_dir,str(i))
        if not os.path.exists(save_dir_cate):
            os.mkdir(save_dir_cate)


    change_count = 0
    #for i in range(cate_num):
    for i in [0,1,2,3,4,5,6,7,8,9]:
        print("=====Start: ",i)


        img_list = getAllName(os.path.join(read_dir, str(i)))
        print("total: ", len(img_list))
        
        for j,img_path in enumerate(img_list):
            if(j%5000==0):
                print("Done: ",j)
                print("change count: ", change_count)
                print("----------")

            img = cv2.imread(img_path)
            if img is None:
                continue

            pre_probs = pre(img)
            acl_score = activaLearning(pre_probs)
            #print("acl_score: ",acl_score)
            pre_cate = np.argmax(pre_probs[0])
            #print(pre_cate)
            if mode == 0:
                if acl_score>1.9:
                    #移动
                    new_path = os.path.join(save_dir, str(pre_cate), os.path.basename(img_path))
                    os.rename(img_path, new_path)
                    change_count+=1

            if mode == 1:
                if acl_score<0.35:
                    #移动
                    new_path = os.path.join(save_dir, str(pre_cate), os.path.basename(img_path))
                    os.rename(img_path, new_path)
                    change_count+=1

            if mode == 2:
                label_cate = i
                if pre_cate!=label_cate:
                    #移动
                    new_path = os.path.join(save_dir, str(pre_cate), os.path.basename(img_path))
                    os.rename(img_path, new_path)
                    change_count+=1
            



if __name__ == '__main__':

    for i in range(1):
        img = cv2.imread("D:/Data/clothes_style/data/TestSet/img_%d.jpg" % i)
        print("img "+str(i)+" pre: "+str(getPreCate(img)))


    # dir_path = "D:/Data/clothes_style/data/DeepFashion/classes7/data_val"
    # calCateAcc(dir_path)


    # read_dir = "D:/Data/clothes_style/data/DeepFashion/classes7/v1_result"
    # save_dir = "D:/Data/clothes_style/data/DeepFashion/classes7/v1_to_cnn"
    # moveNoLabelImg(read_dir, save_dir)
