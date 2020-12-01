import numpy as np
import cv2
import keras
from keras import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Activation,Dropout


class myModel(object):



    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(100,100,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.85))
        self.model.add(Dense(2))
        self.model.add(Activation('sigmoid'))


    def train(self, dataset):
        batch_size = dataset.batch_size
        nb_epoch = dataset.nb_epoch
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model.fit_generator(dataset.train_data_generate(),
                                 steps_per_epoch=dataset.total_train // batch_size,
                                 epochs=nb_epoch,
                                 validation_data=dataset.val_data_generate(),
                                 validation_steps=dataset.total_val//batch_size)


    def save(self, file_path="model.h5"):
        print('Model Saved.')
        self.model.save_weights(file_path)

    def load(self, file_path="model.h5"):
        print('Model Loaded.')
        self.model.load_weights(file_path)

    def predict(self, image):
        # 预测样本分类
        img = image.resize((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        img = image.astype('float32')
        img /= 255

        #归一化
        result = self.model.predict(img)
        print(result)
        # 概率
        result = self.model.predict_classes(img)
        print(result)
        # 0/1

        return result[0]

    def evaluate(self, dataset):
        # 测试样本准确率
        score = self.model.evaluate_generator(dataset.valid,steps=2)
        print("样本准确率%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
