## 基于TF的一些东西

#### 使用inspect_checkpoint来查看ckpt里的内容 打印节点信息
~~~
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.framework import meta_graph
input_graph_def = meta_graph.read_meta_graph_file("model2.ckpt.meta").graph_def
for node in input_graph_def.node:
    print(node.name)
~~~

#### Tensorboard
> tensorboard --logdir=/Users/fire/A

#### pb2onnx
https://github.com/onnx/tensorflow-onnx


#### 剪枝
* [官方示例](https://tensorflow.google.cn/model_optimization/guide/pruning/pruning_with_keras)
* [博客示例](https://www.cnblogs.com/purple5252/p/11812207.html)
* [我的示例(基于mobilenetv2)](./basic/pruned_demo.py)

### Basic

* [TensorFlow Example](./basic/TensorFlowExample.ipynb)
* [graph/ placeholder/ TensorBoard](./basic/Learn_tf.ipynb)
* [模型保存读取](./basic/tf_save_load.ipynb)
* [ckpt转pb](./basic/ckpt2pb.py) | [ckpt模型加载预测](./basic/ckpt_pre.py)
* [pb转tflite](./basic/pb2tflite.py)| [pb模型测试](./basic/tf_pb_pre.py) |  [tf加载pb模型](./basic/read_pb.py)
* [tflite模型测试](./basic/tflite_pre.py) | [转onnx后模型测试](./basic/onnx_pre.py)
* [TFLiteModelMaker轻松利用预训练模型训练tflite(支持efficientnetlite等)](./basic/TFLiteModelMaker)

### Demo

* [逻辑回归](./demo/TF_logsitic.ipynb)
* [mnist手写数字识别(NN)](./demo/basic_mnist_demo.py)
* [mnist手写数字识别(CNN)](./demo/mnist_cnn_demo.py)
* [10人版人脸识别](./demo/ten_people_face_reconize)


### Resource
* [TensorRT安装及使用教程](https://blog.csdn.net/zong596568821xp/article/details/86077553)
