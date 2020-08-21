# Keras 多GPU训练
> Fire 2018.12.05

### 1.指定GPU训练
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
```

### 2.多块GPU训练
直接可以跑，但是通过nvidia-smi观察GPU占用率，只有第一块显卡占用了在跑，其他占用都是闲置的。

设置：

```python

from keras.utils import multi_gpu_model
# 最多支持8块GPU

model = Model(input=.., output=..)  #这里同单卡，声明好模型

parallel_model = multi_gpu_model(model, gpus=4) #这里假设有4块

parallel_model.compile(...) #这里同单卡compile
parallel_model.fit(x, y, epochs=40, batch_size=128)

```

注意：用多卡跑的时候，batchsize要乘以对应的块数，因为会把总的batchsize分到几块上面。

但是还是会报错，说
> could not satisfy explicit device specification '/device:GPU:3' because no supported kernel for GPU device is aviailable

解决方法是，在加载模型之前添加：

``` python
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(allow_soft_placement=True)
set_session(tf.Session(config=config))
```

另外如果遇到如下报错：
```shell
AttributeError: '_TfDeviceCaptureOp' object has no attribute '_set_device_from_string'
```
是由于keras2.2+tensorflow1.14+的一个bug，升级keras到2.3或者降级tensorflow到1.13可以解决。

P.S.网上还有很多人遇到Modelcheckpoint callback报错的问题，我没遇到过，贴一个供参考:

[Keras 多GPU下模型和参数保存Modelcheckpoint callback报错](https://blog.csdn.net/Umi_you/article/details/81301002)


### 3.多块GPU训练模型用多块GPU预测
```python
from keras.utils import multi_gpu_model

basemodel = Model(inputs=input, outputs=y_pred) ##这里同单卡，声明好模型

multi_model=multi_gpu_model(basemodel,gpus=4)
multi_model.load_weights("multi_model.h5") #加载多卡训练的模型

multi_model.predict(...) #预测
```

### 4.多块GPU训练模型用单块GPU预测
多核训练的网络的每一层都是按GPU来命名的，训练时采用多个GPU那么当导入参数的时候必须指定相同数量的GPU才行。所以直接将model切换到单GPU的环境中会报错，此时我们必须将参数保存为单GPU的形式。

```python
from keras.utils import multi_gpu_model

basemodel = Model(inputs=input, outputs=y_pred) ##这里同单卡，声明好模型

multi_model=multi_gpu_model(basemodel,gpus=4)
multi_model.load_weights("multi_model.h5") # 此时basemodel也自动载入了权重，

basemodel.save('basemodel.h5')
```
这里保存的basemodel.h5就是对应单卡的模型，直接在单卡机器上加载就可以使用了。
