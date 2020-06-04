## 基于keras的一些东西

### Common

* #### one-hot
>把数值类标签转为10维向量，标签对应位为1其余为0:
>y_train = keras.utils.np_utils.to_categorical(y_train, 10)

* #### [生成pb](https://www.e-learn.cn/content/wangluowenzhang/193206)
>First, add after your Keras code model.fit(...) and train your model:
```pyhton
from keras import backend as K
import tensorflow as tf
print(model.output.op.name)
saver = tf.train.Saver()
saver.save(K.get_session(), '/tmp/keras_model.ckpt')
```
Then cd to your TensorFlow root directory, run:

python freeze_graph.py --input_meta_graph=./tmp/model0.ckpt.meta --input_checkpoint=./tmp/model0.ckpt --output_graph=./tmp/model0.pb --output_node_names="act_6/truediv" --input_binary=true

>路径：~\anaconda\Lib\site-packages\tensorflow\python\tools

* #### 指定GPU
```pyhton
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

* #### 混合精度训练
```pyhton
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
opt = Adam()
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
# 要把所有模型定义用到的keras全换成tf.keras
```



### Note
* [keras多GPU训练](./note/keras_multiGPU.md)

### Demo

* [Keras Example](./keras_example.ipynb)
* [读取本地mnist数据集_CNN](./demo/keras_mnist.ipynb)
* [RNN分类示例](./demo/RNN_classify.ipynb) | [Keras实现LSTM](./demo/lstm_word_embedding.ipynb) | [LSTM文本生成](./demo/Word_Language_Modelling_LSTM.ipynb)
* [GAN实现mnist图片生成](./demo/Keras_GAN.ipynb)
* [读取本地cifar10数据集_CNN](./demo/keras_cifar10.ipynb)
* [数据增强](./demo/data_aug.py)
* [迭代器读取数据](./demo/data_generator.py)
* [分类baseline](./baseline/) | [预训练模型使用](./demo/pretrain.py) | [预训练模型库](./demo/keras_net.py)
* [cnn热力图可视化](./demo/cam_heatmap.py)
* [h5模型转tflite](./demo/h5_to_tflite.py) | [测试tflite模型结果](./demo/tflite_pre.py) | [有自定义层或函数](https://blog.csdn.net/killfunst/article/details/94295861)
* [带自定义激活函数的h5模型转tflite（比如mobilenetv3的relu6 hard_swish）](./demo/h5_customer_to_tflite.py)
* [训练模型清洗数据（半监督、主动学习）](./demo/clearData.py) 
* [可视化检查自带的数据增强效果](./demo/show_keras_data.py)
* [多输出模型设置class_weight](./demo/multi_output_class_weight.py)
* [用于分类任务的focal loss](./demo/classify_focal_loss.py)
 
### Project
* [双色球预测](./project/Caipiao_nn.ipynb)
* [3D预测（自动获取本地没有的号码）](./project/3D_predict.py)
* [车牌颜色分类](./project/plate_color.ipynb)

### Resource
* [Classification models Zoo](https://github.com/qubvel/classification_models/)
* [EfficientNet](https://github.com/titu1994/keras-efficientnets)
* [mobilenet_v2预训练模型](https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases)
* [Center loss](https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization/blob/master/TYY_mnist.py)
    ```
    #我改写的partial center loss ，针对于背景类不使用centerloss
    l2_loss = Lambda(lambda x: K.prod(K.concatenate([K.sum(K.square(x[0]-x[1][:,0]),axis=1,keepdims=True),x[2]], axis=1), axis=1, keepdims=True),name='l2_loss')([features,centers,input_target])
    ```
* [CV注意力机制CBAM](https://github.com/kobiso/CBAM-keras)
* [优化器lookahead](https://github.com/bojone/keras_lookahead)
* [学习率 SGDR](https://github.com/gunchagarg/learning-rate-techniques-keras)
* [优化器AdaBound](https://github.com/titu1994/keras-adabound)
