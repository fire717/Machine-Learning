## 基于keras的一些东西


* #### one-hot
>把数值类标签转为10维向量，标签对应位为1其余为0:
>y_train = keras.utils.np_utils.to_categorical(y_train, 10)

* #### [生成pb](https://www.e-learn.cn/content/wangluowenzhang/193206)
>First, add after your Keras code model.fit(...) and train your model:
~~~
from keras import backend as K
import tensorflow as tf
print(model.output.op.name)
saver = tf.train.Saver()
saver.save(K.get_session(), '/tmp/keras_model.ckpt')
~~~
Then cd to your TensorFlow root directory, run:

python freeze_graph.py --input_meta_graph=./tmp/model0.ckpt.meta --input_checkpoint=./tmp/model0.ckpt --output_graph=./tmp/model0.pb --output_node_names="act_6/truediv" --input_binary=true

>路径：~\anaconda\Lib\site-packages\tensorflow\python\tools

* #### 指定GPU
~~~
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
~~~




### Note
* [keras多GPU训练](./note/keras_multiGPU.md)

### Demo

* [Keras Example](./keras_example.ipynb)
* [读取本地mnist数据集_CNN](./demo/keras_mnist.ipynb)
* [RNN分类示例](./demo/RNN_classify.ipynb)
* [Keras实现LSTM](./demo/lstm_word_embedding.ipynb)
* [LSTM文本生成](./demo/Word_Language_Modelling_LSTM.ipynb)
* [GAN实现mnist图片生成](./demo/Keras_GAN.ipynb)
* [读取本地cifar10数据集_CNN](./demo/keras_cifar10.ipynb)
* [数据增强](./demo/data_aug.py)
* [迭代器读取数据](./demo/data_generator.py)
* [分类baseline](./baseline/)
* [预训练模型使用](./demo/pretrain.py)
* [cnn热力图可视化](./demo/cam_heatmap.py)
* [h5模型转tflite](./demo/h5_to_tflite.py) | [测试tflite模型结果](./demo/tflite_pre.py) | [有自定义层或函数](https://blog.csdn.net/killfunst/article/details/94295861)
* [训练模型清洗数据（半监督、主动学习）](./demo/clearData.py) 
 
### Project
* [双色球预测](./project/Caipiao_nn.ipynb)
* [3D预测（自动获取本地没有的号码）](./project/3D_predict.py)
* [车牌颜色分类](./project/plate_color.ipynb)
