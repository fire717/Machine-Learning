
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

#### 1.4.3 SSE
* loss = sum|f(x) - Y|^2
* 和方误差，本质同MSE

#### 1.4.4 Smooth L1 Loss (Huber Loss)
* loss = 0.5x^2  if |x|<1, |x|-0.5   otherwise, 导数分别为x和1
* 优点：smooth L1 在x较小时，对x的梯度也会变小，而在x很大时，对x的梯度的绝对值达到上限 1，也不会太大以至于破坏网络参数。smooth L1 loss在 |x| >1的部分采用了 L1 loss，当预测值和目标值差值很大时, 原先L2梯度里的 (f(x) - Y) 被替换成了 ±1,，这样就避免了梯度爆炸, 也就是它更加健壮。完美地避开了 L1 和 L2 损失的缺陷。
* 之所以称为光滑L1函数，是因为此函数处处可导，而原L1函数在x=0处是不可导的。
* 参考
    * [1] [为什么Faster-rcnn、SSD中使用Smooth L1 Loss 而不用Smooth L2 Loss](https://blog.csdn.net/ytusdc/article/details/86659696)


#### 1.4.5 CrossEntoryLoss
* loss = -求和(y' x log(y))
* 为什么分类人物使用CE不用MSE
    * 如果用 MSE 计算 loss，输出的曲线是波动的，有很多局部的极值点。即，非凸优化问题 (non-convex)cross entropy 计算 loss，则依旧是一个凸优化问题[1]
    * 分类标签可以看做是概率分布（由one-hot变换而来），神经网络输出（经过softmax加工）也是一个概率分布，现在想衡量二者的差异（即损失），自然用交叉熵最好了[2]
    * 平均总比有倾向性要好，但这有悖我们的常识;类错误，但偏导为0，权重不会更新，这显然不对——分类越错误越需要对权重进行更新[3]
* 参考
    * [1] [分类模型的 Loss 为什么使用 cross entropy](https://jackon.me/posts/why-use-cross-entropy-error-for-loss-function/)
    * [2] [训练分类器为什么要用交叉熵损失函数而不能用MSE）](https://blog.csdn.net/yhily2008/article/details/80261953)
    * [3] [直观理解为什么分类问题用交叉熵损失而不用均方误差损失?](https://www.cnblogs.com/shine-lee/p/12032066.html)

#### 1.4.6 BCE loss

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
    * [4] [arxiv:Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)

#### 2.1.2 VGG


#### 2.1.3 ResNet

* 引入：VGG网络达到19层后再增加层数就开始导致分类性能的下降，为了解决深层神经网络的难以训练、收敛等问题，提出了残差学习
* 网络结构：ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元
* 残差学习：
    * 普通结构： X -> H(X)，X代表输入，H代表网络层参数计算， H(x)代表学习到的特征
    * 残差结构： F(X) -> H(X)-X，希望其可以学习到残差F(x),这样其实原始的学习特征是H(X)+X,之所以这样是因为残差学习相比原始特征直接学习更容易。 当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。这有点类似与电路中的“短路”，所以是一种短路连接（shortcut connection）。

* 连接方式: 当输入、输出通道数相同时，我们自然可以如此直接使用X进行直接相加。而当它们之间的通道数目不同时，我们就需要考虑建立一种有效的identity mapping函数从而可以使得处理后的输入X与输出Y的通道数目相同即Y = F(X, Wi) + WsxX。当X与Y通道数目不同时，作者尝试了两种方式，一种即简单地将X相对Y缺失的通道直接补零从而使其能够相对齐的方式，另一种则是通过使用1x1的conv来表示Ws映射从而使得最终输入与输出的通道达到一致的方式。the size of the volume does not change within a block[2].
* bottleneck构建模块: 作者提出了一种bottleneck的结构块来代替常规的Resedual block，它像Inception网络那样通过使用1x1 conv来巧妙地缩减或扩张feature map维度从而使得我们的3x3 conv的filters数目不受外界即上一层输入的影响。不过它纯是为了节省计算时间进而缩小整个模型训练所需的时间而设计的，对最终的模型精度并无影响。
* 思考：
    * 为什么残差学习相对更容易：从直观上看残差学习需要学习的内容少，因为残差一般会比较小，学习难度小点。 残差单元的梯度为1+原始梯度，1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易。

* 参考：
    * [1] [你必须要知道CNN模型：ResNet](https://zhuanlan.zhihu.com/p/31852747)
    * [2] [Understanding and visualizing ResNets](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)

#### 2.1.4 Densenet
* 引入：它的基本思路与ResNet一致，但是它建立的是前面所有层与后面层的密集连接（dense connection），它的名称也是由此而来。 DenseNet提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入。
* 连接方式：在DenseNet中，每个层都会与前面所有层在channel维度上连接（concat）在一起（这里各个层的特征图大小是相同的，后面会有说明），并作为下一层的输入。而resnet是元素级相加）。通过特征在channel上的连接来实现特征重用。这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能，这一特点是DenseNet与ResNet最主要的区别。
* 特征图保持一致：CNN网络一般要经过Pooling或者stride>1的Conv来降低特征图的大小，而DenseNet的密集连接方式需要特征图大小保持一致。为了解决这个问题，DenseNet网络中使用DenseBlock+Transition的结构，其中DenseBlock是包含很多层的模块，每个层的特征图大小相同，层与层之间采用密集连接方式。而Transition模块是连接两个相邻的DenseBlock，并且通过Pooling使特征图大小降低。图4给出了DenseNet的网路结构，它共包含4个DenseBlock，各个DenseBlock之间通过Transition连接在一起。
* 非线性组合函数：DenseBlock中的非线性组合函数 采用的是 BN+ReLU+3x3 Conv的结构；由于后面层的输入会非常大，DenseBlock内部可以采用bottleneck层来减少计算量，主要是原有的结构中增加1x1 Conv，如图7所示，即BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv，称为DenseNet-B结构。其中1x1 Conv得到 [公式] 个特征图它起到的作用是降低特征数量，从而提升计算效率。

* 思路：
    * 由于密集连接方式，DenseNet提升了梯度的反向传播，使得网络更容易训练。需要明确一点，dense connectivity 仅仅是在一个dense block里的，不同dense block 之间是没有dense connectivity的
* 参考：
    * [1] [DenseNet：比ResNet更优的CNN模型](https://zhuanlan.zhihu.com/p/37189203)





### 2.2 检测

#### 2.2.1 RCNN

* RCNN
    * 流程：
        1. 给定一张输入图片，从图片中提取 2000 个类别独立的候选区域（包含重叠区域）。
        2. 对于每个区域利用 CNN 抽取一个固定长度的特征向量。
        3. 再对每个区域利用 SVM 进行目标分类。
    * 候选区域：Selective Search 算法[2]
    * 特征提取：resize到227，使用Alexnet提取特征得到4096维的向量
    * 后处理：针对每个类，通过计算IoU指标，采取非极大性抑制，以最高分的区域为基础，剔除掉那些重叠位置的区域。
    * 损失函数：回归使用MSE(L2 loss),使原始框转换到预测框的变换值 和 原始框转换到真实框的变换值 尽可能一样;分类使用交叉熵；两者是独立的
    * 缺点：
        1. 一张图像上有大量的重叠框，所以这些候选框送入神经网络时候，提取特征会有冗余
        2. 训练的空间需求大。因为RCNN中，独立的分类器和回归器需要很多的特征作为训练。RCNN中提取候选框，提取特征和分类回归是分开的，可独立。

* FAST-RCNN
    
    * 特征提取：将整张图片归一化送入神经网络，在最后一层再加入候选框信息（这些候选框还是经过 Selective Search提取，再经过一个ROI层统一映射到最后一层特征图上,而RCNN是通过拉伸来归一化尺寸
    * 损失函数：损失函数使用了多任务损失函数(multi-task loss)，将边框回归直接加入到CNN网络中训练，两者一起训练。回归使用Smooth L1 loss，分类交叉熵。
    * ROI pooling: ROI Pooling 就是将大小不同的feature map 池化成大小相同的feature map，利于输出到下一层网络中。具体是根据输入image，将ROI映射到feature map（ROI）对应位置；将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）；对每个sections进行max pooling操作；
    * 为什么L1替换L2：L1是最小绝对值偏差，是鲁棒的，是因为它能处理数据中的异常值。如果需要考虑任一或全部的异常值，那么最小绝对值偏差是更好的选择。L2范数将误差平方化（如果误差大于1，则误差会放大很多），模型的误差会比L1范数来得大，因此模型会对这个样本更加敏感，这就需要调整模型来最小化误差。如果这个样本是一个异常值，模型就需要调整以适应单个的异常值，这会牺牲许多其它正常的样本，因为这些正常样本的误差比这单个的异常值的误差小。L2是平方差，L1是绝对差，如果有异常点，前者相当于放大了这种误差，而绝对差没有放大。

* FASTER-RCNN
    * faster对fast进行了改进，由于fast rcnn在一张原图上用select search找出2000个候选区域比较耗时，用cpu约耗时2s，为了解决这个问题作者自己设计了一个RPN网络(region proposal network, 10ms每张图像)代替select search。如果把RPN看作一个黑盒子，faster-rnn与fast-rcnn的结构相差不大，只是RPN代替了select search算法，而且select search的位置是在原图中进行的，RPN是在特征图上进行的[6]。
    * RPN网络(region proposal network): RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。(网络最后再进行多分类和回归更精确的框)

* 参考：
    * [1] [【深度学习】R-CNN 论文解读及个人理解](https://blog.csdn.net/briblue/article/details/82012575)
    * [2] [目标检测（1）-Selective Search](https://zhuanlan.zhihu.com/p/27467369)
    * [3] [目标检测1: rcnn流程梳理及边框损失函数loss分析](https://blog.csdn.net/u010397980/article/details/85010554)
    * [4] [Fast_RCNN解读](https://zhuanlan.zhihu.com/p/61611588)
    * [5] [ROI Pooling原理及实现](https://blog.csdn.net/u011436429/article/details/80279536)
    * [6] [目标检测2: faster rcnn对比fast rcnn，训练流程分析，边框损失函数loss分析](https://blog.csdn.net/u010397980/article/details/85055840)

#### 2.2.2 SSD

* 特征层
* 损失函数
    * 回归使用Smooth L1：当预测框与 ground truth 差别过大时，梯度值不至于过大；当预测框与 ground truth 差别很小时，梯度值足够小。
    * One-stage目标检测算法需要同时处理定位和识别的任务，即多任务，其损失函数通常是定位损失和分类损失的加权和

#### 2.2.3 Yolo
* v1
    * 引入：YOLO的核心思想就是利用整张图作为网络的输入，直接在输出层回归bounding box的位置和bounding box所属的类别。没记错的话faster RCNN中也直接用整张图作为输入，但是faster-RCNN整体还是采用了RCNN那种 proposal+classifier的思想，只不过是将提取proposal的步骤放在CNN中实现了。
    * backbone:由GoogLeNet启发而来,有24个卷积层,最后接2个全连接层
    * 实现：
        1. 将一幅图像分成SxS个网格(grid cell)，如果某个object的中心 落在这个网格中，则这个网格就负责预测这个object。
        2. 每个网格要预测B个bounding box(论文中B=2，对应不同的aspect ratio)，每个bounding box除了要回归自身的位置之外，还要附带预测一个confidence值。 这个confidence代表了所预测的box中含有object的置信度和这个box预测的有多准两重信息，其值=Pr(object)xIOU, 其中如果有object落在一个grid cell里，第一项取1，否则取0。 第二项是预测的bounding box和实际的groundtruth之间的IoU值。
        3. 每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类。则SxS个网格，每个网格要预测B个bounding box还要预测C个categories。输出就是S x S x (5xB+C)的一个tensor。 注意：class信息是针对每个网格的，confidence信息是针对每个bounding box的。
    * Loss：yolov1的损失函数全是和方误差SSE，需要理解的是其含义。包含
        * 位置损失：容易理解，负责检测的才有位置损失，其他的都不需回传损失，也就不需要计算，此外小目标对于预测wh的误差更敏感，用先开根再相减的方法缓解。相当于强化了小目标的wh的损失。包括框中心位置x,y损失 + 框宽高w,h损失 
        * confidence损失：负责检测的box的label是在线计算的IOU，不负责和无目标的都为0
        * 类别损失：容易理解，含有目标的网格才有类别损失，其他都不需要回传损失，也就不需要计算。默认网格只出现一种类别，这当然是有缺陷的。yolov1对于一些聚集的目标，检测效果会不好。其实聚集目标本身也算很难检测的情况吧。
        * YOLO并没有使用深度学习常用的均方误差（MSE）而是使用和方误差（SSE）作为损失函数，作者的解释是SSE更好优化。但是SSE作为损失函数时会使模型更倾向于优化输出向量长度更长的任务（也就是分类任务）。为了提升bounding box边界预测的优先级，该任务被赋予了一个超参数coord在论文中=5。
    * 为了解决前、背景样本的样本不平衡的问题，作者给非样本区域的分类任务一个更小的权值noobj在论文中=0.5

* v2
    * backbone:darknet-19(类似vgg)
    * 加入BN: YOLOv2网络通过在每一个卷积层后添加batch normalization，极大的改善了收敛速度同时减少了对其它regularization方法的依赖（舍弃了dropout优化后依然没有过拟合），使得mAP获得了2%的提升。
    * 提高分辨率：YOLO(v1)先以分辨率224x224训练分类网络，然后需要增加分辨率到448x448，这样做不仅切换为检测算法也改变了分辨率。YOLOv2首先修改预训练分类网络的分辨率为448x448，在ImageNet数据集上训练10轮,然后fine tune为检测网络。mAP获得了4%的提升。(说明全卷积支持任意输入，但不一定效果好。)
    * anchor bnox: YOLO(v1)使用全连接层数据进行bounding box预测（要把1470x1的全链接层reshape为7x7x30的最终特征），这会丢失较多的空间信息定位不准。YOLOv2借鉴了Faster R-CNN中的anchor思想： 简单理解为卷积特征图上进行滑窗采样，每个中心预测9种不同大小和比例的建议框。由于都是卷积不需要reshape，很好的保留的空间信息，最终特征图的每个特征点和原图的每个cell一一对应。而且用预测相对偏移（offset）取代直接预测坐标简化了问题，方便网络学习。由anchor box同时预测类别和坐标。
    * K-means聚类: 通过对数据集中的ground true box做聚类，找到ground true box的统计规律。以聚类个数k为anchor boxs个数，以k个聚类中心box的宽高维度为anchor box的维度。通过1-IOU值来度量距离而不是欧式距离，避免大box产生更多loss。
    * passthrough layer： YOLOv2简单添加一个passthrough layer，把浅层特征图（分辨率为26 * 26）连接到深层特征图。passthrough layer把高低分辨率的特征图做连结，叠加相邻特征到不同通道（而非空间位置）类似于Resnet中的identity mappings。这个方法把26x26x512的特征图叠加成13x13x2048的特征图，与原生的深层特征图相连接。本质其实就是特征重排，26x26x512的feature map分别按行和列隔点采样，可以得到4幅13x13x512的特征，把这4张特征按channel串联起来，就是最后的13x13x2048的feature map.还有就是，passthrough layer本身是不学习参数的，直接用前面的层的特征重排后拼接到后面的层，越在网络前面的层，感受野越小，有利于小目标的检测[7]。
    * 多尺度训练：原始YOLO网络使用固定的448x448的图片作为输入，加入anchor boxes后输入变成416x416，由于网络只用到了卷积层和池化层，就可以进行动态调整（检测任意大小图片）。为了让YOLOv2对不同尺寸图片的具有鲁棒性，在训练的时候也考虑了这一点。不同于固定网络输入图片尺寸的方法，每经过10批训练（10 batches）就会随机选择新的图片尺寸。网络使用的降采样参数为32，于是使用32的倍数{320,352，…，608}，最小的尺寸为320x320，最大的尺寸为608x608。 调整网络到相应维度然后继续进行训练。这种机制使得网络可以更好地预测不同尺寸的图片，同一个网络可以进行不同分辨率的检测任务，在小尺寸图片上YOLOv2运行更快，在速度和精度上达到了平衡。
    * 预测的是预测框中心相对于网格单元的偏移量，使用logistic将预测值限制到0-1范围内，这样框偏移就不会超过1个网络（RPN预测anchor box和预测框bbox的偏移量，有可能偏移量很大，导致模型不稳定）
    * YOLO2提出一种联合训练机制，混合来自检测和分类数据集的图像进行训练。当网络看到标记为检测的图像时，基于完整的yolov2损失函数进行反向传播。当它看到一个分类图像时，只从特定于分类的部分反向传播损失[12]。
    * 损失函数同yolo1


* v3
    * backbone: darknet-53 = DarkNet19 + ResNet
    * 网络结构：
        * 结合残差思想，提取更深层次的语义信息。
        * 仍然使用连续的3×3和1×1的卷积层。
        * 通过上采样对三个不同尺度做预测。这样可以预测出更小的物体。
        * 采用了步长为2的卷积层代替pooling层，因为池化层会丢失信息。
        * 整个v3结构里面，是没有池化层和全连接层的。前向传播过程中，张量的尺寸变换是通过改变卷积核的步长来实现的
    * 多标签分类：使用sigmoid和多个logistic,每个框中可能有多个类别物体，而softmax只能用于单分类，因此换成sigmoid，sigmoid可以做多标签分类。
    * 结合不同层特征：类似ssd，这次用了3个特征图，而且在3个特征图上分别预测。同样anchor的大小、比例是根据聚类来获取。大尺度特征图上用小anchor，小尺度特征图上用大anchor。这样可以预测更细粒度的目标。张量拼接将darknet中间层和后面的某一层的上采样进行拼接。拼接的操作和残差层add的操作是不一样的，拼接会扩充张量的维度，而add只是直接相加不会导致张量维度的改变。你说“作者并没有像SSD那样直接采用backbone中间层的处理结果作为feature map的输出，而是和后面网络层的上采样结果进行一个拼接之后的处理结果作为feature map。为什么这么做呢？ 我感觉是有点玄学在里面，一方面避免和其他算法做法重合，另一方面这也许是试验之后并且结果证明更好的选择，再者有可能就是因为这么做比较节省模型size的。这点的数学原理不用去管，知道作者是这么做的就对了” 这个是为了实现小对象检测，因为不同水平的特征图倾向于检测的对象尺寸不同，再者，shallow layer有更详细的定位，而higher layer有更强的判别性特征(分类)，所以作者才这么做的。
    * 采用"leaky ReLU"作为激活函数。基本组件：conv+bn+leakyrelu
    * 损失函数
        * 分类: 对应sigmoid二分类交叉熵BCE loss
        * 回归: 使用MSE
        * 除了w,h的损失函数依然采用总方误差之外，其他部分的损失函数用的是二值交叉熵。最后加到一起

* v4
    * backbone:CSPDarknet53

* 参考：
    * [1] [Yolov1论文](https://arxiv.org/abs/1506.02640)
    * [2] [yolov1 一个cell生成2个bounding box, bounding box 是如何生成的](https://blog.csdn.net/tycoer/article/details/106927119)
    * [3] [【深度学习YOLO V1】深刻解读YOLO V1（图解）](https://blog.csdn.net/c20081052/article/details/80236015)
    * [4] [YOLO v1的详解与复现](https://www.cnblogs.com/xiongzihua/p/9315183.html)
    * [5] [物体检测之YOLO](https://zhuanlan.zhihu.com/p/42772125?from=groupmessage)
    * [6] [YOLO2](https://zhuanlan.zhihu.com/p/25167153)
    * [7] [如何理解YOLOv2中的passthrough layer？](https://www.zhihu.com/question/58903330)
    * [8] [Yolov2论文](https://arxiv.org/abs/1612.08242)
    * [9] [YOLO v2 详解](https://www.pianshen.com/article/7293987192/#passthrough__126)
    * [10] [YOLO1-5的BackBone和Neck](https://blog.csdn.net/qq_35447659/article/details/108121631)
    * [11] [yolo系列之yolo v3【深度解析】](https://blog.csdn.net/leviopku/article/details/82660381)
    * [12] [YOLO1、YOLO2、YOLO3对比](https://blog.csdn.net/qq_32172681/article/details/100104494)
    * [13] [目标检测算法之YOLOv2损失函数详解](https://zhuanlan.zhihu.com/p/93632171)

### 2.3 分割
#### 2.3.1 U-net

## 四、优化、实现

卷积一般通过一种im2col方式实现

wingrad卷积

## 五、应用

### 4.1 OCR
#### 4.1.1 CRNN

#### 4.1.2 CPTN

## 六、其他

#### 6.1 特征图直接elementwise相加和concat有什么区别
elementwise相加相当于人工提取特征，而且可能丢失信息；concat让模型去学习特征。但是concate带来的计算量较大，在明确原始特征的关系可以使用add操作融合的话，使用add操作可以节省计算代价。
* 参考：
    * [1] [神经网络中通过add和concate（cat）的方式融合特征的不同](https://blog.csdn.net/weixin_42926076/article/details/100660188)
    * [2] [如何理解神经网络中通过add的方式融合特征？](https://www.zhihu.com/question/306213462/answer/562776112)