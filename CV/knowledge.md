
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
    
* 思考：
    * 对于CNN，BN的操作是在各个特征维度之间单独进行，也就是说各个通道是分别进行Batch Normalization操作的。
    * 为什么需要乘以gamma加上beta：
        * 说法一：不同层的值域是可能不一样的，而BN让使输出变大变小这个重要工作更容易做到[3],;
        * 说法二：这是为了让神经网络自己去学着使用和修改这个扩展参数 gamma, 和 平移参数 β, 这样神经网络就能自己慢慢琢磨出前面的 normalization 操作到底有没有起到优化的作用, 如果没有起到作用, 我就使用 gamma 和 belt 来抵消一些 normalization 的操作[4]
        * 说法三：这两个参数的引入是为了恢复数据本身的表达能力(直接归一化会导致网络表达能力下降)，对规范化后的数据进行线性变换，通过自适应学习让数据保留更多的原始信息[5]
    * 为什么要在激活函数前：原始论文BN是在激活函数前的，一种解释是对于输入100和1000，经过激活函数后的值，比如sigmoid（1/(1+np.exp(-x))）后都为1了，再规范化已经没有意义。但是又有人发现实际中放在后面更好[7].
    * 缺点：依赖于batch的大小，当batch值很小时，计算的均值和方差不稳定。研究表明对于ResNet类模型在ImageNet数据集上，batch从16降低到8时开始有非常明显的性能下降
    * 引申：BN、LN、IN、GN和SN [9]

* 参考：
    * [1] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    * [2] [BatchNormalization的理解](https://blog.csdn.net/qq_34823656/article/details/96431067)
    * [3] [深度学习中 Batch Normalization是什么，效果为什么好？](https://zhuanlan.zhihu.com/p/140330586)
    * [4] [什么是批标准化 (Batch Normalization)](https://zhuanlan.zhihu.com/p/24810318?utm_source=qq)
    * [5] [批量归一化(BatchNormalization)](https://zhuanlan.zhihu.com/p/108837487)
    * [6] [深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762/answer/607815171)
    * [7] [Batch-normalized 应该放在非线性激活层的前面还是后面？](https://www.zhihu.com/question/283715823/answer/438882036) 
    * [8] [论文|How Does Batch Normalizetion Help Optimization](https://zhuanlan.zhihu.com/p/66683061)
    * [9] [深度学习中的五种归一化（BN、LN、IN、GN和SN）方法简介](https://blog.csdn.net/u013289254/article/details/99690730)



#### 1.1.2 Dropout
* 是什么：Dropout是一种在深度学习环境中应用的正则化手段（一般作为一个网络层），为了缓解模型过拟合的问题
* 实现
    * 训练时：每个神经单元以概率p被保留(dropout丢弃率为1-p)，丢失的即直接对神经元输出乘以0
    * 测试时：权重参数w要乘以p，输出是：pw，相当于把输出缩放到同一范围
    * 反向Dropout(Inverted Dropout)：在训练阶段缩放激活函数1/p，从而使得其测试阶段保持不变。更常用

* 为什么有效
    * 引入随机噪声
    * 丢失神经元强制模型学习更多特征
    * 类似bagging的模型融合[3]
    * 减少神经元之间复杂的共适应关系[2]

* 参考：
    * [1] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    * [2] [Dropout原理与实现](https://www.cnblogs.com/zingp/p/11631913.html)
    * [3] [bagging与dropout的异同](https://blog.csdn.net/daizongxue/article/details/79123134)

* 思考：
    * 引申：DropConnect

#### Attention

### 1.2 激活函数
#### Sigmoid

#### Tanh

#### Relu

#### Softmax

### 1.3 优化器

#### SGD

#### Adam

#### Ranger

### 1.4 损失函数

#### 1.4.1 L1 loss (MAE)
* loss = |f(x) - Y|, 导数为 正负f'(x)
* 缺点：L1 对 x 的导数为常数。这就导致训练后期，预测值与 ground truth 差异很小时， L1 损失对预测值的导数的绝对值仍然为 1，而 learning rate 如果不变，损失函数将在稳定值附近波动，难以继续收敛以达到更高精度。相比L2对异常点没那么敏感。

#### 1.4.2 L2 loss (MSE)
* loss = |f(x) - Y|^2, 导数为 2(f(x)-Y)f'(x)
* 缺点：当 x 增大时 L2 损失对 x 的导数也增大。这就导致训练初期，预测值与 groud truth 差异过于大时，损失函数对预测值的梯度十分大，训练不稳定。从下面的形式 L2 Loss的梯度包含 (f(x) - Y)，当预测值 f(x) 与目标值 Y 相差很大时（此时可能是离群点、异常值(outliers)），容易产生梯度爆炸

#### 1.4.3 Smooth L1 Loss (Huber Loss)
* loss = 0.5x^2  if |x|<1, |x|-0.5   otherwise, 导数分别为x和1
* 优点：smooth L1 在x较小时，对x的梯度也会变小，而在x很大时，对x的梯度的绝对值达到上限 1，也不会太大以至于破坏网络参数。smooth L1 loss在 |x| >1的部分采用了 L1 loss，当预测值和目标值差值很大时, 原先L2梯度里的 (f(x) - Y) 被替换成了 ±1,，这样就避免了梯度爆炸, 也就是它更加健壮。完美地避开了 L1 和 L2 损失的缺陷。
* 之所以称为光滑L1函数，是因为此函数处处可导，而原L1函数在x=0处是不可导的。
* 参考
    * [1] [为什么Faster-rcnn、SSD中使用Smooth L1 Loss 而不用Smooth L2 Loss](https://blog.csdn.net/ytusdc/article/details/86659696)


#### CrossEntoryLoss
* loss = -求和(y' x log(y))
* 为什么分类人物使用CE不用MSE
    * 如果用 MSE 计算 loss，输出的曲线是波动的，有很多局部的极值点。即，非凸优化问题 (non-convex)cross entropy 计算 loss，则依旧是一个凸优化问题[1]
    * 分类标签可以看做是概率分布（由one-hot变换而来），神经网络输出（经过softmax加工）也是一个概率分布，现在想衡量二者的差异（即损失），自然用交叉熵最好了[2]
    * 平均总比有倾向性要好，但这有悖我们的常识;类错误，但偏导为0，权重不会更新，这显然不对——分类越错误越需要对权重进行更新[3]
* 参考
    * [1] [分类模型的 Loss 为什么使用 cross entropy](https://jackon.me/posts/why-use-cross-entropy-error-for-loss-function/)
    * [2] [训练分类器为什么要用交叉熵损失函数而不能用MSE）](https://blog.csdn.net/yhily2008/article/details/80261953)
    * [3] [直观理解为什么分类问题用交叉熵损失而不用均方误差损失?](https://www.cnblogs.com/shine-lee/p/12032066.html)

#### BCE los

#### focal-loss
#### CTC-Loss


## 三、网络

### 2.1 分类
#### 2.1.1 MobileNet
* v1: 
    * 基本单元是深度可分离卷积（depthwise separable convolution）,基本结构是3x3 depthwise Conv - BN - Relu - 1x1 conv - BN -Relu
    * 网络结构首先是一个3x3的标准卷积，然后后面就是堆积depthwise separable convolution，并且可以看到其中的部分depthwise convolution会通过strides=2进行down sampling。然后采用average pooling将feature变成1x1，根据预测类别大小加上全连接层，最后是一个softmax层；
    * 整个计算量基本集中在1x1卷积上，如果你熟悉卷积底层实现的话，你应该知道卷积一般通过一种im2col方式实现，其需要内存重组，但是当卷积核为1x1时，其实就不需要这种操作了，底层可以有更快的实现。对于参数也主要集中在1x1卷积，除此之外还有就是全连接层占了一部分参数。
    * 引入了两个模型超参数：width multiplier和resolution multiplier。第一个参数width multiplier主要是按比例减少通道数，其取值范围为(0,1]。第二个参数resolution multiplier主要是按比例降低特征图的大小，resolution multiplier仅仅影响计算量，但是不改变参数量。


* v2:
    * 改进1：Inverted residuals，通常的residuals block是先经过一个1 * 1的Conv layer，把feature map的通道数“压”下来，再经过3 * 3 Conv layer，最后经过一个1 * 1 的Conv layer，将feature map 通道数再“扩张”回去。即先“压缩”，最后“扩张”回去。 而inverted residuals就是先“扩张”，最后“压缩”。因为若是采用以往的residual block，先“压缩”，再卷积提特征，那么DWConv layer可提取得特征就太少了，因此一开始不“压缩”，MobileNetV2反其道而行，一开始先“扩张”，本文实验“扩张”倍数为6。
    * 改进2：Linear bottlenecks，为了避免Relu对特征的破坏，在residual block的Eltwise sum之前的那个 1 * 1 Conv 不再采用Relu。因为当采用“扩张”→“卷积提特征”→ “压缩”时，在“压缩”之后Relu对于负的输入，输出全为零，会破坏特征；而本来特征就已经被“压缩”，再经过Relu的话，又要“损失”一部分特征，因此这里不采用Relu。
    * 基本组件bottleneck：1×1 conv - BN - Relu6 - 3x3 dwConv - BN - Relu6 - 1x1 conv
    * 除了最后的avgpool，整个网络并没有采用pooling进行下采样，而是利用stride=2来下采样，此法已经成为主流，不知道是否pooling层对速度有影响，因此舍弃pooling层?
    * 相比v1准确率提升，参数量减少，推理耗时减少。

* v3
    * 没有引入新的 Block，使用神经架构搜索来搜索结构
    * 搜索结果使用了MnasNet 模型引入的基于squeeze and excitation结构的轻量级注意力模型
    * 在网络结构搜索中，作者结合两种技术：资源受限的NAS（platform-aware NAS）与NetAdapt，前者用于在计算和参数量受限的前提下搜索网络的各个模块，所以称之为模块级的搜索（Block-wise Search） ，后者用于对各个模块确定之后网络层的微调。
    * 使用激活函数h-swish，作为swish的数值近似（swish计算量较大），h-swish(X) = X x Relu6(X+3)/6
    * 作者们发现MobileNetV2 网络端部最后阶段的计算量很大，重新设计了这一部分
    * mobilenetv3-small相比v2准确率提升，参数量减少，推理耗时减少。

* 深度可分离卷积:
    * 传统卷积： 输入(224,224,3) ，使用(3x3x3)的卷积核5个, 输出为(224,224,5) （假设padding），计算量为3x3x3x5x224x224 = 6773760
    * 深度可分离卷积: 输入(224,224,3) ，DepthwiseConv使用(3x3)的卷积核3个,得到(224,224,3), 然后PointwiseConv使用(1x1)的卷积核5个，得到(224,224,5)，计算量为3x3x3x224x224+1x1x3x5x224x224 = 2107392，缩小比例为 (3x3x3x5x224x224)/(3x3x3x224x224+1x1x3x5x224x224) = (3x3x5)/(3x3+5) = 3.2倍, 取决于原始卷积核的大小和输出的通道数

* 参考：
    * [1] [【深度学习MobileNet】——深刻解读MobileNet网络结构](https://blog.csdn.net/c20081052/article/details/80703896)
    * [2] [轻量化网络：MobileNet-V2](https://blog.csdn.net/u011995719/article/details/79135818)
    * [3] [重磅！MobileNetV3 来了！](https://www.jiqizhixin.com/articles/2019-05-09-2)

#### VGG

#### ResNet

### 2.2 检测

#### 2.2.1 FastRCNN
RCNN, FAST-RCNN, FASTER-RCNN的发展历史
* 损失函数
    * 回归使用Smooth L1：当预测框与 ground truth 差别过大时，梯度值不至于过大；当预测框与 ground truth 差别很小时，梯度值足够小。

#### 2.2.2 SSD

* 特征层
* 损失函数
    * 回归使用Smooth L1：当预测框与 ground truth 差别过大时，梯度值不至于过大；当预测框与 ground truth 差别很小时，梯度值足够小。
    *  One-stage目标检测算法需要同时处理定位和识别的任务，即多任务，其损失函数通常是定位损失和分类损失的加权和

#### 2.2.3 Yolo
* 损失函数
    * 回归边框使用MSE



## 四、优化

卷积一般通过一种im2col方式实现

wingrad卷积

## 五、应用

### 4.1 OCR
#### 4.1.1 CRNN

#### 4.1.2 CPTN
