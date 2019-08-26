## 基于TF的一些东西

#### 使用inspect_checkpoint来查看ckpt里的内容 打印节点信息
~~~
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.framework import meta_graph
input_graph_def = meta_graph.read_meta_graph_file("model2.ckpt.meta").graph_def
for node in input_graph_def.node:
    print(node.name)
~~~

#### tensorboard
> tensorboard --logdir=/Users/fire/A

### 基础

* [TensorFlow Example](./basic/TensorFlowExample.ipynb)
* [graph/ placeholder/ TensorBoard](./basic/Learn_tf.ipynb)
* [模型保存读取](./basic/tf_save_load.ipynb)
* [tf加载pb模型](./basic/read_pb.py)
* [pb转tflite](./basic/pb2tflite.py)
* [pb模型测试](./basic/tf_pb_pre.py)
* [tflite模型测试](./basic/tflite_pre.py)

### Demo

* [逻辑回归](./demo/TF_logsitic.ipynb)
* [mnist手写数字识别(NN)](./demo/basic_mnist_demo.py)
* [mnist手写数字识别(CNN)](./demo/mnist_cnn_demo.py)
* [10人版人脸识别](./demo/ten_people_face_reconize)


