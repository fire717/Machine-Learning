
import os
import json
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import load_model
import numpy as np
import random
import math
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.optimizers import Adam,SGD

random.seed(2020)
np.random.seed(2020)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

def generate(batch, shape, ptrain, pval):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
        ptrain: train dir.
        pval: eval dir.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        channel_shift_range=20,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=30,
        width_shift_range=0.05,
        height_shift_range=0.05)
        

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical',
        shuffle=True)

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical',
        shuffle=True)

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2



def myGenerator(raw_generator):
    while True:
        x,y = raw_generator.next()

        label_smoothing = 0.1
        y = y * (1 - label_smoothing) + label_smoothing / 2

        yield [x,y]

def getGeneratorImgs(raw_generator, iters):
    imgs = []
    for i in range(iters):
        x,y = raw_generator.next()
        imgs.extend(x)
    return np.array(imgs)

def train(cfg):
    
    epochs = cfg['epochs']
    save_dir = cfg['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    shape = (int(cfg['height']), int(cfg['width']), 3)

    n_class = int(cfg['class_number'])
    batch_size = int(cfg['batch_size'])


    if cfg['model'] == 'mymodel':
        from model.my_model import MyModel
        model = MyModel(shape, n_class).build()

    if cfg['model'] == 'v2':
        from model.mobilenet_v2 import MyModel
        model = MyModel(shape, n_class).buildRaw()


    train_generator, validation_generator, count1, count2 = generate(batch_size, shape[:2], cfg['train_dir'], cfg['eval_dir'])
    print(count1, count2)


    earlystop = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(filepath=os.path.join("save", 'prune_e_{epoch:02d}_{val_loss:.3f}_{val_acc:.3f}.h5'),
                 monitor='val_acc', save_best_only=False, save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=1e-7)



    model_path = r'./save/v2'

    

    # x_train, y_train = train_generator.next()
    # num_train_samples = batch_size
    # x_test, y_test = validation_generator.next()

    
    loaded_model = tf.keras.models.load_model(os.path.join(model_path,'e_06_0.20_1.00.h5'))
    score = loaded_model.evaluate_generator(validation_generator, count2//batch_size)
    print('original Test loss:', score[0])
    print('original Test accuracy:', score[1])


    
    end_step = np.ceil(1.0 * count1 / batch_size).astype(np.int32) * epochs
    print(end_step)
    new_pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50, 
                                                                        final_sparsity=0.90,  
                                                                        begin_step=0,
                                                                        end_step=end_step,                                                         
                                                                        frequency=100)}
    new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)

    #new_pruned_model.summary()
    opt = Adam(lr=float(0.0001))
    new_pruned_model.compile(loss=tf.keras.losses.categorical_crossentropy,     
                            optimizer=opt,     
                            metrics=['acc'])
    #现在我们开始训练和修剪模型。

    #Add a pruning step callback to peg the pruning step to the optimizer's
    #step. Also add a callback to add pruning summaries to tensorboard
    logdir = "./save/log"
    callbacks = [earlystop,checkpoint,reduce_lr,
                sparsity.UpdatePruningStep(),    
                sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)]
    # new_pruned_model.fit(x_train, y_train,          
    #                 batch_size=batch_size,          
    #                 epochs=epochs,          
    #                 verbose=1,          
    #                 callbacks=callbacks,          
    #                 validation_data=(x_test, y_test))

    new_pruned_model.fit_generator(train_generator, 
            validation_data=validation_generator, 
            steps_per_epoch=100,#count1 // batch_size,
            validation_steps=count2 // batch_size,
            epochs=epochs,
            callbacks=callbacks)

    score = new_pruned_model.evaluate_generator(validation_generator, count2//batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    final_model = sparsity.strip_pruning(new_pruned_model)

    new_pruned_keras_file = "save/pruned_model.h5"
    tf.keras.models.save_model(final_model, new_pruned_keras_file, include_optimizer=False)




if __name__ == '__main__':
    # with open('config/config.json', 'r') as f:
    #     cfg = json.load(f)

    cfg = {
    "model": "v2",
    "height": 224,
    "width": 224,
    "class_number": 2,
    "batch_size": 64,
    "epochs": 10,
    "train_dir": "/home/AlgorithmicGroup/yw/workshop/antiface/data/v7/train",
    "eval_dir": "/home/AlgorithmicGroup/yw/workshop/antiface/data/v7/val",
    "save_dir": "save",
    "weights": ""
    }
    train(cfg)
    #nohup python -u train_cls.py > nohup.log 2>&1 &
