#coding = utf-8
#Fire / 2018.6.30
import requests
from bs4 import BeautifulSoup as bs
import re
import time

import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras.layers import Dropout

#玄学参数 随机种子
win_number = 2018
from numpy.random import seed
seed(win_number)
from tensorflow import set_random_seed
set_random_seed(win_number)
          
class predict3D(object):
    """docstring for predict3D"""
    def __init__(self, series='-1', histotyFile='history3D.txt'):
        self.series = series
        self.histotyFile = histotyFile
        self.data = None #最新的数据在最后 str格式
        self.latest = -1 #str

        self.model1 = None
        self.model2 = None
        self.model3 = None
        
        self.ball1_train = None
        self.ball1_label = None
        self.ball2_train = None
        self.ball2_label = None
        self.ball3_train = None
        self.ball3_label = None

        self.seq_N = 10
        self.batch_size = 1024
        self.epochs = 100
        self.validation_split = 0.01
        self.hidden_size = 64
          
    def readHistory(self):
        with open(self.histotyFile,'r') as f:
            raw_data = f.readlines()

        self.latest = raw_data[0].split(':')[0]
        #print(int(''.join(raw_data[0].split(':')[1].split(' ')[:-1])))
        data = []
        for line in raw_data:
            data.append(''.join(line.split(':')[1].split(' ')[:-1]))
        self.data = data[::-1] #最新的数据在最后
        print('Finish read local data: ',len(self.data))

    def getLatestNumber(self):
        url = "https://datachart.500.com/sd/history/inc/history.php?limit=18173&start=2000000&end=2018173"
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, sdch',
            'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Host': 'datachart.500.com',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        }
        r = requests.get(url=url, headers=headers)
        r.encoding = 'gb2312'
        html = r.text
        # 解析html
        soup = bs(html, 'lxml')
        ul = soup.find_all(name='tr',attrs={'class': 't_tr1'})
        latest = str(ul[0].find_next(name='td'))[4:11]
        if latest == self.latest:
            print("本地已经是最新数据")
        else:
            print("本地数据数: ", len(self.data))
            dist = int(latest) - int(self.latest)
            newdata = []
            for i in range(dist):
                newdata.append(''.join(str(ul[i].find_next(name='td',attrs={'class': 'cfont2'})).split('>')[1][:5].split(' ')))
            newdata = newdata[::-1]
            self.data = self.data + newdata    
            print("添加后本地数据数: ", len(self.data))

    def buildModel(self, input_size = 10, seq_len = None, hidden_size=None):
        if not seq_len:
            seq_len = self.seq_N
        if not hidden_size:
            hidden_size = self.hidden_size
        model = Sequential()
        model.add(LSTM(hidden_size, input_shape=( seq_len,input_size)))
        model.add(Dropout(0.2))
        model.add(Dense(input_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        #print(model.summary())
        return model

    def makeModelData(self):
        self.readHistory()
        self.getLatestNumber()

        seq_N = self.seq_N
        ball1_train = []
        ball1_label = []
        ball2_train = []
        ball2_label = []
        ball3_train = []
        ball3_label = []
        for i in range(seq_N,len(self.data)):
            data_tmp = self.data[i-seq_N:i][::-1] ## 小trick。逆序

            x_tmp = [int(x[0]) for x in data_tmp] 
            x_tmp = keras.utils.np_utils.to_categorical(x_tmp, 10)
            ball1_train.append(x_tmp)
            
            y_tmp = int(self.data[i][0])
            y_tmp = keras.utils.np_utils.to_categorical(y_tmp, 10)
            ball1_label.append(y_tmp)

            x_tmp = [int(x[1]) for x in data_tmp] 
            x_tmp = keras.utils.np_utils.to_categorical(x_tmp, 10)
            ball2_train.append(x_tmp)
            
            y_tmp = int(self.data[i][1])
            y_tmp = keras.utils.np_utils.to_categorical(y_tmp, 10)
            ball2_label.append(y_tmp)

            x_tmp = [int(x[2]) for x in data_tmp]  
            x_tmp = keras.utils.np_utils.to_categorical(x_tmp, 10)
            ball3_train.append(x_tmp)
            
            y_tmp = int(self.data[i][2])
            y_tmp = keras.utils.np_utils.to_categorical(y_tmp, 10)
            ball3_label.append(y_tmp)
            

        ball1_train = np.array(ball1_train)
        ball1_label = np.array(ball1_label)
        ball1_label = np.reshape(ball1_label,(ball1_label.shape[0],ball1_label.shape[2]))

        ball2_train = np.array(ball2_train)
        ball2_label = np.array(ball2_label)
        ball2_label = np.reshape(ball2_label,(ball2_label.shape[0],ball2_label.shape[2]))

        ball3_train = np.array(ball3_train)
        ball3_label = np.array(ball3_label)
        ball3_label = np.reshape(ball3_label,(ball3_label.shape[0],ball3_label.shape[2]))

        self.ball1_train = ball1_train
        self.ball1_label = ball1_label
        self.ball2_train = ball2_train
        self.ball2_label = ball2_label
        self.ball3_train = ball3_train
        self.ball3_label = ball3_label
        print("Make data finished.And ball 1 data lenth: ",ball1_train.shape, ball1_label.shape)

    def train(self):
        self.makeModelData()
        self.model1 = self.buildModel()
        self.model1.fit(self.ball1_train, self.ball1_label,batch_size=self.batch_size,epochs=self.epochs,validation_split=self.validation_split)

        self.model2 = self.buildModel()
        self.model2.fit(self.ball2_train, self.ball2_label,batch_size=self.batch_size,epochs=self.epochs,validation_split=self.validation_split)

        self.model3 = self.buildModel()
        self.model3.fit(self.ball3_train, self.ball3_label,batch_size=self.batch_size,epochs=self.epochs,validation_split=self.validation_split)

        print('Finish train.')

    def predict(self):
        data_tmp = self.data[-self.seq_N:][::-1] ## 小trick。逆序

        true_last1 = [int(x[0]) for x in data_tmp] 
        true_last1 = keras.utils.np_utils.to_categorical(true_last1, 10)
        test_x1 = np.reshape(true_last1, (1, self.seq_N, 10))
        test_y1 = self.model1.predict(test_x1)
        test_y1 = np.argmax(test_y1)

        true_last2 = [int(x[1]) for x in data_tmp] 
        true_last2 = keras.utils.np_utils.to_categorical(true_last2, 10)
        test_x2 = np.reshape(true_last2, (1, self.seq_N, 10))
        test_y2 = self.model2.predict(test_x2)
        test_y2 = np.argmax(test_y2)

        true_last3 = [int(x[2]) for x in data_tmp] 
        true_last3 = keras.utils.np_utils.to_categorical(true_last3, 10)
        test_x3 = np.reshape(true_last3, (1, self.seq_N, 10))
        test_y3 = self.model3.predict(test_x3)
        test_y3 = np.argmax(test_y3)

        print("上期期数：",self.latest, "上期号码：", self.data[-1])
        
        pre = 0
        print("本期预测号码：", test_y1, test_y2, test_y3)

model = predict3D()
model.train()
model.predict()
