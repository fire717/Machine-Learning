
## CV知识点


## 一、基础

### 1.1 通用

#### 1.1.1 BatchNormalization

* BN是什么: Normalization即数据归一化，将数据按比例缩放，使之落入一个小的特定区间（[0,1]），可用于不同单位或量级的指标能够进行比较和加权。BatchNormalization就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。

* 实现: 
    * 训练时
        1. 求一个batch内数据的均值和方差
        2. 通过减均值除以方差的操作把数据归一化
        3. 对归一化后的数据乘以gamma加上beta得到最终数据, gamma和beta是需要学习的参数
    * 预测时使用的就是所有训练实例中获得的统计量来代替Mini-Batch里面m个训练实例获得的均值和方差统计量，因为本来就打算用全局的统计量，只是因为计算量等太大所以才会用Mini-Batch这种简化方式的，那么在推理的时候直接用全局统计量即可

* 为什么需要: 
    * 论文[1]中提出内部协变量移位（internal covariate shift）的问题，即ML系统实例集合<X,Y>中的输入值X的分布老是变，这不符合IID独立同分布假设，网络模型很难稳定的学规律；但论文中对工作原理的解释在2018年被MIT的研究人员推翻说后面又证明数学推导是错的[8]
    * 不同层输入输出可能数据分布不一致，导致不同特征值可能映射后的结果差不多，导致特征无法学习
    * 随着网络加深，训练越来越困难，收敛越来越慢，且容易出现梯度爆炸、消失
    * 让loss函数更光滑[8]

* 优点：
    * 加速训练（归一化后使得后一层网络不必不断去适应底层网络中输入的变化，实现了网络中层与层之间的解耦，允许每一层进行独立学习，有利于提高整个神经网络的学习速度[5]），同时因为更稳定，可以使用更大的学习率
    * 使得模型对网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定[5]
    * 允许网络使用饱和性激活函数（例如sigmoid，tanh等），缓解梯度消失问题[5]
    * 具有一定的正则化效果，由于我们使用mini-batch的均值与方差作为对整体训练样本均值与方差的估计，尽管每一个batch中的数据都是从总体样本中抽样得到，但不同mini-batch的均值与方差会有所不同，这就为网络的学习过程中增加了随机噪音，与Dropout通过关闭神经元给网络训练带来噪音类似，在一定程度上对模型起到了正则化的效果[5],增加泛化能力
    
* 一些思考：
    * 对于CNN，BN的操作是在各个特征维度之间单独进行，也就是说各个通道是分别进行Batch Normalization操作的。
    * 为什么需要乘以gamma加上beta：
        * 说法一：不同层的值域是可能不一样的，而BN让使输出变大变小这个重要工作更容易做到[3],;
        * 说法二：这是为了让神经网络自己去学着使用和修改这个扩展参数 gamma, 和 平移参数 β, 这样神经网络就能自己慢慢琢磨出前面的 normalization 操作到底有没有起到优化的作用, 如果没有起到作用, 我就使用 gamma 和 belt 来抵消一些 normalization 的操作[4]
        * 说法三：这两个参数的引入是为了恢复数据本身的表达能力(直接归一化会导致网络表达能力下降)，对规范化后的数据进行线性变换，通过自适应学习让数据保留更多的原始信息[5]
    * 为什么要在激活函数前：原始论文BN是在激活函数前的，一种解释是对于输入100和1000，经过激活函数后的值，比如sigmoid（1/(1+np.exp(-x))）后都为1了，再规范化已经没有意义。但是又有人发现实际中放在后面更好[7].
    * 缺点：依赖于batch的大小，当batch值很小时，计算的均值和方差不稳定。研究表明对于ResNet类模型在ImageNet数据集上，batch从16降低到8时开始有非常明显的性能下降

* 参考：
    * [1] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    * [2] [BatchNormalization的理解](https://blog.csdn.net/qq_34823656/article/details/96431067)
    * [3] [深度学习中 Batch Normalization是什么，效果为什么好？](https://zhuanlan.zhihu.com/p/140330586)
    * [4] [什么是批标准化 (Batch Normalization)](https://zhuanlan.zhihu.com/p/24810318?utm_source=qq)
    * [5] [批量归一化(BatchNormalization)](https://zhuanlan.zhihu.com/p/108837487)
    * [6] [深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762/answer/607815171)
    * [7] [Batch-normalized 应该放在非线性激活层的前面还是后面？](https://www.zhihu.com/question/283715823/answer/438882036) 
    * [8] [论文|How Does Batch Normalizetion Help Optimization](https://zhuanlan.zhihu.com/p/66683061)



#### 1.1.2 Dropout




### 1.2 激活函数

### 1.3 优化器

### 1.4 损失函数

#### CTC-Loss


## 三、网络

### 2.1 分类
#### 2.1.1 MobileNet
* v1
* v2
* v3
* 深度可分离卷积

#### 

### 2.2 检测

#### 2.2.1 SSD

* 特征层
* 损失函数

#### 2.2.2 Yolo

#### 2.2.3 FastRCNN

## 四、应用

### 4.1 OCR
#### 4.1.1 CRNN

#### 4.1.2 CPTN
