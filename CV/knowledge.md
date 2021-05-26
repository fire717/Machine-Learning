
# CV知识点


## 目录
* 一、基础 
* 二、数据
* 三、网络 
* 四、实现
* 五、应用
* 六、其他

## 一、基础

### 1.1 通用

#### 1.1.1 BatchNormalization

* BN是什么: Normalization即数据归一化，将数据按比例缩放，使之落入一个小的特定区间（[0,1]），可用于不同单位或量级的指标能够进行比较和加权。BatchNormalization就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。

* 实现: 
    * 训练时
        1. 求一个channel内数据的均值和方差
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
    * 对于CNN，BN的操作是在各个特征维度之间单独进行，也就是说各个通道是分别进行Batch Normalization操作的。对每个batch中的每个channel中的所有值进行单独处理，最终可以得到C个mean与C个var的值，在训练的过程中，BN是首先计算当前channel中所有值得均值与方差，然后对其进行归一化。为什么？我觉得可以这么理解：首先bn是跟在cnn之后的，拿到的就是ncwh的数据维度，而这个c就是对应不同的c个卷积核，对应于全连接层的每一个神经元，然后对每一个卷积核（神经元），求nhw维度（对全连接就是n）的归一化。对于不同图片、同一图片的不同位置，生成一个channel的卷积核是同一个，所以针对非channel其他维度做归一化,从而保证每一个卷积核提取特征的输出数值稳定性。
    * 我们在一些源码中，可以看到带有BN的卷积层，bias设置为False，就是因为即便卷积之后加上了Bias，在BN中也是要减去的，所以加Bias带来的非线性就被BN一定程度上抵消了。需要补偿。
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
    * [10] [BatchNorm的个人解读和Pytorch中BN的源码解析](https://blog.csdn.net/qq_34914551/article/details/102736271)


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

#### 1.1.3 Spatial Pyramid Pooling
* 原理：
    具体地,在一个CNN里,把最以后一次池化层去掉,换成一个SPP去做最大池化操作(max pooling).如果最后一次卷积得到了k个feature map,也就是有k个filter,SPP有M个bin,那经过SPP得到的是一个kM维的向量.我的理解是,比如上图中第一个feature map有16个bin,一共有256个feature map,每一个经过16个bin的max pooling得到16个数,那256个feature map就是16x256的向量了.SPP的bin大小可以选择多个,所以经过SPP还能产生4x256,1x256维的向量.假设原图输入是224x224，对于conv5出来后的输出是13x13x256的，可以理解成有256个这样的filter，每个filter对应一张13x13的reponse map。如果像上图那样将reponse map分成1x1(金字塔底座)，2x2(金字塔中间)，4x4（金字塔顶座）三张子图，分别做max pooling后，出来的特征就是(16+4+1)x256 维度。如果原图的输入不是224x224，出来的特征依然是(16+4+1)x256维度。这样就实现了不管图像尺寸如何 池化n 的输出永远是 （16+4+1）x256 维度。实际运用中只需要根据全连接层的输入维度要求设计好空间金字塔即可。
* 思考：
    * 卷积神经网络中，当输入不是固定size时，如何解决
        * 方案1：对输入进行resize，统一到同一大小。
        * 方案2：取消全连接层，对最后的卷积层global average polling（GAP。
        * 方案3：在第一个全连接层前，加入SPP layer。本文要介绍的。
        * p.s.以上方案还要实测，具体哪种方案比较好，强烈推荐方案2，3。

* 参考：
    * [1] [Spatial Pyramid Pooling讲解](https://zhuanlan.zhihu.com/p/34788333?utm_source=ZHShareTargetIDMore)
    * [2] [Spatial Pyramid Pooling 详解](https://www.jianshu.com/p/e36aef9b7a8a)


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
* 与KL散度的联系[4]
* 参考
    * [1] [分类模型的 Loss 为什么使用 cross entropy](https://jackon.me/posts/why-use-cross-entropy-error-for-loss-function/)
    * [2] [训练分类器为什么要用交叉熵损失函数而不能用MSE）](https://blog.csdn.net/yhily2008/article/details/80261953)
    * [3] [直观理解为什么分类问题用交叉熵损失而不用均方误差损失?](https://www.cnblogs.com/shine-lee/p/12032066.html)
    * [4] [交叉熵、相对熵（KL散度）、JS散度和Wasserstein距离（推土机距离）](https://zhuanlan.zhihu.com/p/74075915)

#### 1.4.6 BCE loss
* BCELoss 是CrossEntropyLoss的一个特例，只用于二分类问题，而CrossEntropyLoss可以用于二分类，也可以用于多分类。，在使用nn.BCELoss需要在该层前面加上Sigmoid函数


#### 1.4.7 Focal-loss
* Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。
* 实现：
    * 首先在原有的基础上加了一个因子，其中gamma>0使得减少易分类样本的损失。使得更关注于困难的、错分的样本。例如gamma为2，对于正类样本而言，预测结果为0.95肯定是简单样本，所以（1-0.95）的gamma次方就会很小，这时损失函数值就变得更小。而预测概率为0.3的样本其损失相对很大。对于负类样本而言同样，预测0.1的结果应当远比预测0.7的样本损失值要小得多。对于预测概率为0.5时，损失只减少了0.25倍，所以更加关注于这种难以区分的样本。这样减少了简单样本的影响，大量预测概率很小的样本叠加起来后的效应才可能比较有效。
    * 此外，加入平衡因子alpha，用来平衡正负样本本身的比例不均：文中alpha取0.25，即正样本要比负样本占比小，这是因为负例易分。
* 思考：
    *  focalloss原始论文是针对目标检测的前景和背景两类  迁移到多分类要修改一些地方。主要就是原始是(1-y')^gamma, 改成abs(one_hot_label-y'))^gamma
* 参考：
    * [1] [Focal Loss理解](https://www.cnblogs.com/king-lps/p/9497836.html)

#### 1.4.8 Center-loss
* Center-loss搭配交叉熵使用，在结合使用这两种损失函数时，可以认为softmax交叉熵负责增加类间距离，center-loss负责减小类内距离，这样学习到的特征判别度会更高。
* 实现：
    * 当前数据减去该batch的中心数据特征的二范求和
    * 没法直接获得c,所以将其放到网络里自己生成,在每一个batch里更新center.即随机初始化center,而后每一个batch里计算当前数据与center的距离,而后将这个梯度形式的距离加到center上.类似于参数修正.同样的类似于梯度下降法,这里再增加一个scale度量a,使得center不会抖动.
* 参考：
    * [1] [Center-Loss](https://blog.csdn.net/wxb1553725576/article/details/80602786)
    * [2] [中心损失 Center Loss 解释](https://www.cnblogs.com/carlber/p/10811396.html)
    * [3] [github一个pytorch实现](https://github.com/egcode/facerec/blob/master/losses/CenterLoss.py)

#### 1.4.9 Contrastive Loss
* 输入是一个数据对x1,x2和标签y,y=1代表相似，0代表不相似
* 当y=1时，loss相当于L2 loss，y=0时，对L2设置了一个阈值ｍargin，表示我们只考虑不相似特征欧式距离在０～ｍargin之间的，当距离超过ｍargin的，则把其loss看做为０
* 参考：
    * [1] [孪生神经网络(Siamese Network)详解](https://blog.csdn.net/weixin_45250844/article/details/102765678)

#### 1.4.10 IoU Loss
* 主要是改进目标检测中边框回归任务的loss，比如ssd用的smooth-l1，但是他们都是把坐标角点单独去优化忽略了相关性，iou-loss就是直接通过iou来计算loss。
* 参考：
    * [1] [IOU & GIOU & DIOU 介绍及其代码实现](https://blog.csdn.net/leonardohaig/article/details/103394369)

#### 1.4.10 Wing Loss
* loss = wln(1+|x|/epsilon)  if |x|<w, |x|-C   otherwise,   (|x|为标签和预测的差异，一般来说就是x_label-x_pre)。论文中实验结果为w=10，epsilon=2时效果最后。
    * w：非线性区域的长度
    * epsilon：非线性区域的曲率（值越大曲线越弯）
    * C： C = w - wln(1+w/epsilon)（保证连续）
* 用于人脸关键点回归。回归一般用L1，L2和Smooth L1，但是他们在-1,1这个区域都倾向于被large error主导，而关键点检测任务不像检测框回归，关键点误差很小，所以要更好的关注small error，所以使用ln函数,梯度为1/x，则x越小梯度越大。论文说，在同时组合大量点后，梯度由small error主导，步长由large error主导，刚好可以平衡。这个步长不好理解，论文说L1和L2两个函数的梯度大小分别为1和|x|，对应的优化步长大小为|x|和1，而ln(x)梯度为1/x，步长为x^2,文章[2]说步长是需要优化的次数，那就是|x|/梯度，那不应该叫步长，应该叫迭代次数，步长有误导.而为了避免在错误的方向上“走”一大步，因此在small errors 时，需要gradient进行限制，本文就采用两个参数来控制gradient，分别是w和epsilon。
* 参考：
    * [1] [人脸关键点: Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks](https://blog.csdn.net/u011995719/article/details/80150508)
    * [2] [[人脸关键点检测] Wing loss 论文解读](https://blog.csdn.net/john_bh/article/details/106302026)
    
#### 1.4.12 KL散度

#### 1.4.13 CTC-Loss


## 二、数据

### 2.1 数据增强

#### 2.1.1 MixUp
* 实现1：
    1. 根据参数alpha的值，得到Beta分布的随机值gamma
    2. 设原始输入数据batch和标签label（类别值，不需要为one-hot）
    3. mixed_batch1 = batch，label1 = label
    4. mixed_batch2 = shuffle(batch)，label2 = label对应shuffle后的标签
    5. input_batch = gamma * mixed_batch1 + (1-gamma) * mixed_batch2
    6. new_loss = gamma * loss_function(input_batch, label1) + (1 - gamma) * loss_function(input_batch, label2)
* 实现2：
    1. 根据参数alpha的值，得到Beta分布的随机值gamma
    2. 设原始输入数据batch和标签label（需要为one-hot）
    3. mixed_batch1 = batch，label1 = label
    4. mixed_batch2 = shuffle(batch)，label2 = label对应shuffle后的标签
    5. input_batch = gamma * mixed_batch1 + (1-gamma) * mixed_batch2
    6. input_label = gamma * label1 + (1-gamma) * label2
    7. new_loss = loss_function(input_batch, input_label)
* 原始论文是对两个不同batch进行融合，一般在代码实现过程中，两个batch图片是同一批样本，唯一不同的是，batch1是原始batch图片样本，而b atch2是对batch1在batch size维度进行shuffle后得到的[1]
* 思考:
    实现1是论文中的提到的开源代码的实现方式，实现2是论文中示例代码的实现。经过大致演算推导，可以从2推到1，两者应该是一样的。但是在我自己的二分类任务训练过程中train-loss，val-loss不是完全一样的，实现2的train-loss更低，且训练完实现2的val-acc更高，而test集上两者acc一样，实现1的F1值稍高。
* 参考
    * [1] [数据增强之mixup算法详解](https://blog.csdn.net/sinat_36618660/article/details/101633504)
    * [2] [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
    * [3] [facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py)


## 三、网络

### 3.1 分类
#### 3.1.1 MobileNet 
* v1 (2017)
    * 基本单元是深度可分离卷积（depthwise separable convolution）,基本结构是3x3 depthwise Conv - BN - Relu - 1x1 conv - BN -Relu
    * 网络结构首先是一个3x3的标准卷积，然后后面就是堆积depthwise separable convolution，并且可以看到其中的部分depthwise convolution会通过strides=2进行down sampling。然后采用average pooling将feature变成1x1，根据预测类别大小加上全连接层，最后是一个softmax层；整个网络共28层，特征图宽度逐渐降低，深度逐渐变深。
    * 整个计算量基本集中在1x1卷积上，如果你熟悉卷积底层实现的话，你应该知道卷积一般通过GEMM实现，底层是im2col方式实现，其需要内存重组（内存不连续需要多次访问内存），但是当卷积核为1x1时，其实就不需要这种操作了（内存连续），底层可以有更快的实现（Caffe 在计算卷积时，首先用 im2col 将输入的三维数据转换成二维矩阵，使得卷积计算可表示成两个二维矩阵相乘，从而充分利用已经优化好的 GEMM 库来为各个平台加速卷积计算）。对于参数也主要集中在1x1卷积，除此之外还有就是全连接层占了一部分参数。
    * 引入了两个模型超参数：width multiplier和resolution multiplier。第一个参数width multiplier主要是按比例减少通道数，其取值范围为(0,1]。第二个参数resolution multiplier主要是按比例降低特征图的大小，resolution multiplier仅仅影响计算量，但是不改变参数量。
    * 除了最后的avgpool，整个网络并没有采用pooling进行下采样，而是利用stride=2来下采样，此法已经成为主流，不知道是否pooling层对速度有影响，因此舍弃pooling层.
    *在 MobileNet V1 里面使用 ReLU6，ReLU6 就是普通的ReLU但是限制最大输出值为 6，这是为了在移动端设备 float16/int8 的低精度的时候，也能有很好的数值分辨率，如果对 ReLU 的激活范围不加限制，输出范围为0到正无穷，如果激活值非常大，分布在一个很大的范围内，则低精度的float16/int8无法很好地精确描述如此大范围的数值，带来精度损失。
    * 缺点：MobileNet V1 的结构其实非常简单，论文里是一个非常复古的直筒结构，类似于VGG一样。这种结构的性价比其实不高，后续一系列的 ResNet, DenseNet 等结构已经证明通过复用图像特征，使用 Concat/Eltwise+ 等操作进行融合，能极大提升网络的性价比

* v2 (2018)
    * 改进1：基本单元由普通的深度可分离卷积变为Inverted residuals block。首先引入了残差结构，但通常的residuals block是先经过一个1 * 1的Conv layer，把feature map的通道数“压”下来，再经过3 * 3 Conv layer，最后经过一个1 * 1 的Conv layer，将feature map 通道数再“扩张”回去。即先“压缩”，最后“扩张”回去。 而inverted residuals就是先“扩张”，最后“压缩”。因为若是采用以往的residual block，先“压缩”，再卷积提特征，那么DWConv layer可提取得特征就太少了，因此一开始不“压缩”，MobileNetV2反其道而行，一开始先“扩张”，本文实验“扩张”倍数为6。因为引入了残差结构，虽然有扩张，但是提取特征能力更强，可以使用更小的输入、输出维度。
    * 改进2：Linear bottlenecks，为了避免Relu对特征的破坏，在residual block的Eltwise sum之前的那个 1 * 1 Conv 不再采用Relu。因为当采用“扩张”→“卷积提特征”→ “压缩”时，在“压缩”之后Relu对于负的输入，输出全为零，会破坏特征；而本来特征就已经被“压缩”，再经过Relu的话，又要“损失”一部分特征，因此这里不采用Relu。
    * 基本组件bottleneck：1×1 conv - BN - Relu6 - 3x3 dwConv - BN - Relu6 - 1x1 conv。整体网络依然先是一个普通conv，然后一个bottleneck扩张1倍，后续6个bottleneck都是扩展6倍（这6个bottleneck重复次数分别是234331），然后一个1x1conv把通道变为1280，然后GAP，再1x1conv变为输出维度。
    * 相比v1准确率提升，参数量减少，推理耗时减少。

* v3 (2019)
    * 没有引入新的 Block，使用神经架构搜索来搜索结构
    	* 在网络结构搜索中，作者结合两种技术：资源受限的NAS（platform-aware NAS）与NetAdapt，前者用于在计算和参数量受限的前提下搜索网络的各个模块，所以称之为模块级的搜索（Block-wise Search） ，后者用于对各个模块确定之后网络层的微调。
		* NetAdapt作用：用户可以自动简化一个预训练的网络以使其达到硬件资源限制，同时最大化精确度。 NetAdapt简介：将 direct metrics（延时，能量，内存占用等等， 等等，或者是这些指标的结合）并入自适应算法，direct metrics 用empirical measurements （实证测量）分析，这样就不用对特殊平台的细节进行了解了（当然将来的改进可以对平台细节进行了解）。在每次迭代中，NetAdapt会差生很多network proposal并将他们在目标平台上测量，以测量结果指导NetAdapt产生下一批network proposal。
	* 引入了基于squeeze and excitation结构的轻量级注意力模型。SE模块加入再bottleneck的3x3depthwise卷积后的bn和激活函数之间。（SE结构：pool-fc-relu-fc-hsigmoid，且fc深度变为原始的1/4，这样可以提升速度）。对于SE模块，不再使用sigmoid，而是采用ReLU6(x + 3) / 6作为近似（就像h-swish那样）。
    * 使用激活函数h-swish替换relu6，作为swish的数值近似（swish计算量较大），h-swish(X) = X x Relu6(X+3)/6.但是只是高层用了hswish，底层还是relu。
    * 常规的swish使用的是 x x sigmoid(x)，它可以显著提高神经网络的精度，但是sigmoid的计算实在是太耗时了，所以在这里作者使用了ReLU6作为替代。不过，并非整个模型都使用了h-swish，模型的前一半层使用常规ReLU（第一个conv层之后的除外）。 因为作者发现，h-swish仅在更深层次上有用。 此外，考虑到特征图在较浅的层中往往更大，因此计算其激活成本更高，所以作者选择在这些层上简单地使用ReLU（而非ReLU6），因为它比h-swish省时。具体解释一下如何完成ReLU6(x + 3) / 6的。如图2所示，在Mul层中，做了乘以0.16667的乘法，这就相当于除以6；ReLU6则融合在了卷积层之中；另外，对于x+3，这里的3被加在了卷积层的偏置层中了。这样做也是一种小的优化方式。
    * 作者们发现MobileNetV2 网络端部最后阶段的计算量很大，重新设计了这一部分
    * mobilenetv3-small相比v2准确率提升，参数量减少，推理耗时减少。
    * MobileNet v1和v2都从具有32个滤波器的常规3×3卷积层开始，然而实验表明，这是一个相对耗时的层，只要16个滤波器就足够完成对224 x 224特征图的滤波。虽然这样并没有节省很多参数，但确实可以提高速度。
    * v3-small的基本结构是一个3x3x16全卷积，然后3个3x3的bneck，使用relu，然后8个5x5的bneck，使用hswish。然后1x1-pool-1x1-1x1  第二个1x1对应mb2中的1280，移到了pool后提速。
    * v3高层都是5x5conv，且扩张倍数不是6，貌似是搜出来的，理论依据暂没搜到。

* 深度可分离卷积:
    * 传统卷积： 输入(224,224,3) ，使用(3x3x3)的卷积核5个, 输出为(224,224,5) （假设padding），计算量为3x3x3x5x224x224 = 6773760
    * 深度可分离卷积: 输入(224,224,3) ，DepthwiseConv使用(3x3)的卷积核3个,得到(224,224,3), 然后PointwiseConv使用(1x1)的卷积核5个，得到(224,224,5)，计算量为3x3x3x224x224+1x1x3x5x224x224 = 2107392，缩小比例为 (3x3x3x5x224x224)/(3x3x3x224x224+1x1x3x5x224x224) = (3x3x5)/(3x3+5) = 3.2倍, 取决于原始卷积核的大小和输出的通道数

* 参考：
    * [1] [【深度学习MobileNet】——深刻解读MobileNet网络结构](https://blog.csdn.net/c20081052/article/details/80703896)
    * [2] [轻量化网络：MobileNet-V2](https://blog.csdn.net/u011995719/article/details/79135818)
    * [3] [重磅！MobileNetV3 来了！](https://www.jiqizhixin.com/articles/2019-05-09-2)
    * [4] [arxiv:Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)

#### 3.1.2 VGG (2014)
* 将5x5和7x7这样的卷积分解为多个3x3的卷积的堆叠
* 以VGG16为例，输入(224,224,3),包含5个block，最后接3个全连接层(4096-4096-1000),pool(2*2, s=2)
    * block1: conv(3x3)x64-conv(3x3)x64-maxpool , 输出112x112x64
    * block2: conv(3x3)x128-conv(3x3)x128-maxpool , 输出56x56x128
    * block3: conv(3x3)x256-conv(3x3)x256-conv(1x1)x256-maxpool , 输出28x28x256
    * block4: conv(3x3)x512-conv(3x3)x512-conv(1x1)x512-maxpool , 输出14x14x512
    * block5: conv(3x3)x512-conv(3x3)x512-conv(1x1)x512-maxpool , 输出7x7x512

* 参考：
    * [1] [卷积神经网络VGG16详解](https://baijiahao.baidu.com/s?id=1667221544796169037&wfr=spider&for=pc)
    * [2] [vgg16介绍](https://blog.csdn.net/how0723/article/details/83059277)

#### 3.1.3 GoogLeNet(Inception)

* GoogLeNet原始版本 (2014)
    * GoogLeNet相比于之前的卷积神经网络的最大改进是设计了一个稀疏参数的网络结构，但是能够产生稠密的数据，既能增加神经网络表现，又能保证计算资源的使用效率。
    * 网络结构：
        * 基本结构：对前一层，并行拼接[1x1conv,3x3conv,5x5conv,3x3maxpool]
            1. 采用不同大小的卷积核意味着不同大小的感受野，最后拼接意味着不同尺度特征的融合
            2. 之所以卷积核大小采用1、3和5，主要是为了方便对齐。设定卷积步长stride=1之后，只要分别设定pad=0、1、2，那么卷积之后便可以得到相同维度的特征，然后这些特征就可以直接拼接在一起了 
            3. 文章说很多地方都表明pooling挺有效，所以Inception里面也嵌入了
            4. 网络越到后面，特征越抽象，而且每个特征所涉及的感受野也更大了，因此随着层数的增加，3x3和5x5卷积的比例也要增加

* Inception V1 (2014)
    * 一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，这也就意味着巨量的参数，容易产生过拟合也会大大增加计算量。解决上述两个缺点的根本方法是将全连接甚至一般的卷积都转化为稀疏连接。一方面现实生物神经系统的连接也是稀疏的，另一方面有文献表明：对于大规模稀疏的神经网络，可以通过分析激活值的统计特性和对高度相关的输出进行聚类来逐层构建出一个最优网络。这点表明臃肿的稀疏网络可能被不失性能地简化。 早些的时候，为了打破网络对称性和提高学习能力，传统的网络都使用了随机稀疏连接。但是，计算机软硬件对非均匀稀疏数据的计算效率很差，所以在AlexNet中又重新启用了全连接层，目的是为了更好地优化并行运算。为了既能保持网络结构的稀疏性，又能利用密集矩阵的高计算性能。大量的文献表明可以将稀疏矩阵聚类为较为密集的子矩阵来提高计算性能，据此论文提出了名为Inception 的结构来实现此目的。
    * Inception 结构的主要思路是怎样用密集成分来近似最优的局部稀疏结构。
    * 网络结构：
        * 原始GoogLeNet使用5x5的卷积核仍然会带来巨大的计算量。 此借鉴NIN2，采用1x1卷积核来进行降维,即在3x3conv和5x5conv前和3x3maxpool后分别加一个1x1conv
        * 例如：上一层的输出为100x100x128，经过具有256个输出的5x5卷积层之后(stride=1，pad=2)，输出数据为100x100x256。其中，卷积层的参数为128x5x5x256。假如上一层输出先经过具有32个输出的1x1卷积层，再经过具有256个输出的5x5卷积层，那么最终的输出数据仍为为100x100x256，但卷积参数量已经减少为128x1x1x32 + 32x5x5x256，大约减少了4倍。
        * 网络最后采用了GAP来代替全连接层，想法来自NIN,事实证明可以将TOP1 accuracy提高0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便以后大家finetune
        * 为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度。文章中说这两个辅助的分类器的loss应该加一个衰减系数，但看caffe中的model也没有加任何衰减。此外，实际测试的时候，这两个额外的softmax会被去掉。
        * GoogleNet的caffemodel大小约50M
    * GoogLeNet V1出现的同期，性能与之接近的大概只有VGGNet了，并且二者在图像分类之外的很多领域都得到了成功的应用。但是相比之下，GoogLeNet的计算效率明显高于VGGNet，大约只有500万参数，只相当于Alexnet的1/12(GoogLeNet的caffemodel大约50M，VGGNet的caffemodel则要超过600M)。
* Inception V2 (2015)
    * 通过简单地放大Inception结构来构建更大的网络，则会立即提高计算消耗。文章中作者首先给出了一些已经被证明有效的用于放大网络的通用准则和优化方法。这些准则和方法适用但不局限于Inception结构（这些并不能直接用来提高网络质量，而仅用来在大环境下作指导。）：
        1. 避免表达瓶颈，特别是在网络靠前的地方。 信息流前向传播过程中显然不能经过高度压缩的层，即表达瓶颈。从input到output，feature map的宽和高基本都会逐渐变小，但是不能一下子就变得很小。比如你上来就来个kernel = 7, stride = 5 ,这样显然不合适。 
        2. 高维特征更易处理。 高维特征更易区分，会加快训练。
        3. 可以在低维嵌入上进行空间汇聚而无需担心丢失很多信息。 比如在进行3x3卷积之前，可以对输入先进行降维而不会产生严重的后果。假设信息可以被简单压缩，那么训练就会加快。
        4. 平衡网络的宽度与深度。
    * 大尺寸的卷积核可以带来更大的感受野，但也意味着更多的参数，比如5x5卷积核参数是3x3卷积核的25/9=2.78倍。为此，作者提出可以用2个连续的3x3卷积层(stride=1)组成的小网络来代替单个的5x5卷积层，(保持感受野范围的同时又减少了参数量)【原来这里提出的】。这种替代不会造成表达缺失，3x3卷积之后还要再加激活吗？实验表明添加非线性激活会提高性能。
    * 大卷积核完全可以由一系列的3x3卷积核来替代，那能不能分解的更小一点呢。文章考虑了 nx1 卷积核。任意nxn的卷积都可以通过1xn卷积后接nx1卷积来替代。实际上，作者发现在网络的前期使用这种分解效果并不好，还有在中度大小的feature map上使用效果才会更好。（对于mxm大小的feature map,建议m在12到20之间）。该结构被正式用在GoogLeNet V2中，即把3x3变成1x3和3x1的两层。另外输出的维度channel，一般来说会逐渐增多(每层的num_output)，否则网络会很难训练。（特征维度并不代表信息的多少，只是作为一种估计的手段）
* Inception V3 (2015)
    * Inception Net v3 整合了前面 Inception v2 中提到的所有升级，还使用了：
        * RMSProp 优化器；
        * Factorized 7x7 卷积；
        * 辅助分类器使用了 BatchNorm；
        * 标签平滑
    * Inception V3一个最重要的改进是卷积分解（Factorization），将7x7卷积分解成两个一维的卷积串联（1x7和7x1），3x3卷积分解为两个一维的卷积串联（1x3和3x1），这样既可以加速计算，又可使网络深度进一步增加，增加了网络的非线性（每增加一层都要进行ReLU）。
* Inception V4 (2016)
    * Inception v4 和 Inception -ResNet 在同一篇论文《Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning》中介绍。
    * inception v4把原来的inception结构中加入了ResNet中的Residual Blocks结构，把一些层的输出加上前几层的输出，这样中间这几层学习的实际上是残差。论文说引入ResNet中的Residual Blocks结构不是用来提高准确度，只是用来提高模型训练收敛速度。
    * 另外就是V4把一个先1x1卷积再3x3卷积换成了先3x3卷积再1x1卷积。
    * 受 ResNet 的优越性能启发，研究者提出了一种混合 inception 模块。Inception ResNet 有两个子版本：v1 和 v2，Inception-ResNet v1 的计算成本和 Inception v3 的接近。Inception-ResNetv2 的计算成本和 Inception v4 的接近。它们有不同的 stem，两个子版本都有相同的模块 A、B、C 和缩减块结构。唯一的不同在于超参数设置。


* 参考：
    * [1] [GoogLeNet系列解读](https://blog.csdn.net/shuzfan/article/details/50738394)
    * [2] [从Inception v1到Inception-ResNet，一文概览Inception家族的「奋斗史」 ](https://www.sohu.com/a/233423803_129720)
    * [3] [GoogLeNet和Inception v1、v2、v3、v4网络介绍](https://blog.csdn.net/zgcr654321/article/details/90264871)


#### 3.1.4 ResNet (2016)

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

#### 3.1.5 Densenet (2017)
* 引入：它的基本思路与ResNet一致，但是它建立的是前面所有层与后面层的密集连接（dense connection），它的名称也是由此而来。 DenseNet提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入。
* 连接方式：在DenseNet中，每个层都会与前面所有层在channel维度上连接（concat）在一起（这里各个层的特征图大小是相同的，后面会有说明），并作为下一层的输入。而resnet是元素级相加。通过特征在channel上的连接来实现特征重用。这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能，这一特点是DenseNet与ResNet最主要的区别。
* 特征图保持一致：CNN网络一般要经过Pooling或者stride>1的Conv来降低特征图的大小，而DenseNet的密集连接方式需要特征图大小保持一致。为了解决这个问题，DenseNet网络中使用DenseBlock+Transition的结构，其中DenseBlock是包含很多层的模块，每个层的特征图大小相同，层与层之间采用密集连接方式。而Transition模块是连接两个相邻的DenseBlock，并且通过Pooling使特征图大小降低。图4给出了DenseNet的网路结构，它共包含4个DenseBlock，各个DenseBlock之间通过Transition连接在一起。
* 非线性组合函数：DenseBlock中的非线性组合函数 采用的是 BN+ReLU+3x3 Conv的结构；由于后面层的输入会非常大，DenseBlock内部可以采用bottleneck层来减少计算量，主要是原有的结构中增加1x1 Conv，如图7所示，即BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv，称为DenseNet-B结构。其中1x1 Conv得到 [公式] 个特征图它起到的作用是降低特征数量，从而提升计算效率。

* 思考：
    * 由于密集连接方式，DenseNet提升了梯度的反向传播，使得网络更容易训练。需要明确一点，dense connectivity仅仅是在一个dense block里的，不同dense block 之间是没有dense connectivity的
    * 特征图直接elementwise相加和concat有什么区别
        elementwise相加相当于人工提取特征，而且可能丢失信息；concat让模型去学习特征。但是concate带来的计算量较大，在明确原始特征的关系可以使用add操作融合的话，使用add操作可以节省计算代价。

* 参考：
    * [1] [DenseNet：比ResNet更优的CNN模型](https://zhuanlan.zhihu.com/p/37189203)
    * [2] [神经网络中通过add和concate（cat）的方式融合特征的不同](https://blog.csdn.net/weixin_42926076/article/details/100660188)
    * [3] [如何理解神经网络中通过add的方式融合特征？](https://www.zhihu.com/question/306213462/answer/562776112)


#### 3.1.1 ShuffleNet 
* v1 (2017)
    * ShuffleNet的结构基本上和ResNet是一样的，也是分成几个stage（ResNet中有4个stage，这里只有3个），然后在每个stage中用ShuffleNet unit代替原来的Residual block，这也就是ShuffleNet算法的核心。
    * 基于ResNet残差单元修改，引入channel shuffle、pointwise group convolutions和depthwise separable convolution。
    * 网络结构首先是一个3x3的标准卷积（s=2），然后是maxpooling（s=2），后面就是3个stage（block），最后一个GAP转全连接输出。
    * 原始残差：x -> 1x1conv(bn+relu) -> 3x3DWconv(bn+relu) -> 1x1pconv(bn) -> add(x)(relu) 
    * 改进1: x -> 1x1Gconv(bn+relu) -> channel shuffle -> 3x3DWconv(bn) -> 1x1Gconv(bn) -> add(x)(relu)
    * 改进2：x -> 1x1Gconv(bn+relu) -> channel shuffle -> 3x3DWconv(bn),s=2 -> 1x1Gconv(bn) -> concat( 3x3AVGpool(x),s=2 )(relu)
    * 思考
    	* 核心就是认为1x1卷积计算多，于是引入分组pointwise卷积，而为了解决分组后每个部分关注区域有限的问题，引入了channel shuffle 
        * 为什么block后面没有relu：同mobilenet2吧，小网络再加就没啥特征了。
        * 为什么add改为concat: concat相比add可以提升复杂度，add类似于先验知识两个特征图位置对应可以add，而concat则是让模型自己学习，原文说的是很小的计算代价提升模型精度。同时因为有channel shuffle，所以不是一一对应了？所以用concat更好。
        * 为什么改进2要步长2且引入avg？看三个block的图是先引入avg，然后为了保持一致，所以DWconv 的s设为2，同时为了不丢失太多信息，add改为concat

* v2 (2018)
    * 改进1：


* 参考：
    * [1] [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
    * [2] [ShuffleNet算法详解](https://blog.csdn.net/u014380165/article/details/75137111)
    * [3] [轻量化网络：ShuffleNet](https://blog.csdn.net/u011995719/article/details/78918128)
    * [4] []()


### 3.2 检测

#### 3.2.0 基础
* 目标检测为什么不能直接回归框坐标：[One Stage目标检测算法可以直接回归目标的坐标是什么原理？ - 叶不知的回答 - 知乎](https://www.zhihu.com/question/304307091/answer/544905898)

#### 3.2.1 RCNN

* RCNN
    * 流程：
        1. 给定一张输入图片，从图片中提取 2000 个类别独立的候选区域（包含重叠区域）。
        2. 对于每个区域利用 CNN 抽取一个固定长度的特征向量。
        3. 再对每个区域利用 SVM 进行目标分类。
    * 候选区域：Selective Search 算法[2]
    * 特征提取：resize到227，使用Alexnet提取特征得到4096维的向量
    * 后处理：针对每个类，通过计算IoU指标，采取非极大性抑制，以最高分的区域为基础，剔除掉那些重叠位置的区域。
    * 损失函数：回归使用MSE(L2 loss),使原始框转换到预测框的变换值 和 原始框转换到真实框的变换值 尽可能一样（为什么 不直接回归真实坐标见[7]）;分类使用交叉熵；两者是独立的
    * 缺点：
        1. 一张图像上有大量的重叠框，所以这些候选框送入神经网络时候，提取特征会有冗余
        2. 训练的空间需求大。因为RCNN中，独立的分类器和回归器需要很多的特征作为训练。RCNN中提取候选框，提取特征和分类回归是分开的，可独立。

* Fast-RCNN
    
    * 特征提取：将整张图片归一化送入神经网络，在最后一层再加入候选框信息（这些候选框还是经过 Selective Search提取，再经过一个ROI层统一映射到最后一层特征图上,而RCNN是通过拉伸来归一化尺寸
    * 损失函数：损失函数使用了多任务损失函数(multi-task loss)，将边框回归直接加入到CNN网络中训练，两者一起训练。回归使用Smooth L1 loss，分类交叉熵。
    * ROI pooling: ROI Pooling 就是将大小不同的feature map 池化成大小相同的feature map，利于输出到下一层网络中。具体是根据输入image，将ROI映射到feature map（ROI）对应位置；将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）；对每个sections进行max pooling操作；
    * 为什么L1替换L2：L1是最小绝对值偏差，是鲁棒的，是因为它能处理数据中的异常值。如果需要考虑任一或全部的异常值，那么最小绝对值偏差是更好的选择。L2范数将误差平方化（如果误差大于1，则误差会放大很多），模型的误差会比L1范数来得大，因此模型会对这个样本更加敏感，这就需要调整模型来最小化误差。如果这个样本是一个异常值，模型就需要调整以适应单个的异常值，这会牺牲许多其它正常的样本，因为这些正常样本的误差比这单个的异常值的误差小。L2是平方差，L1是绝对差，如果有异常点，前者相当于放大了这种误差，而绝对差没有放大。

* Faster-RCNN
    * faster对fast进行了改进，由于fast rcnn在一张原图上用select search找出2000个候选区域比较耗时，用cpu约耗时2s，为了解决这个问题作者自己设计了一个RPN网络(region proposal network, 10ms每张图像)代替select search。如果把RPN看作一个黑盒子，faster-rnn与fast-rcnn的结构相差不大，只是RPN代替了select search算法，而且select search的位置是在原图中进行的，RPN是在特征图上进行的[6]。
    * RPN网络(region proposal network): RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。(网络最后再进行多分类和回归更精确的框)

* 参考：
    * [1] [【深度学习】R-CNN 论文解读及个人理解](https://blog.csdn.net/briblue/article/details/82012575)
    * [2] [目标检测（1）-Selective Search](https://zhuanlan.zhihu.com/p/27467369)
    * [3] [目标检测1: rcnn流程梳理及边框损失函数loss分析](https://blog.csdn.net/u010397980/article/details/85010554)
    * [4] [Fast_RCNN解读](https://zhuanlan.zhihu.com/p/61611588)
    * [5] [ROI Pooling原理及实现](https://blog.csdn.net/u011436429/article/details/80279536)
    * [6] [目标检测2: faster rcnn对比fast rcnn，训练流程分析，边框损失函数loss分析](https://blog.csdn.net/u010397980/article/details/85055840)
    * [7] [（杂）解释 为什么目标检测不直接预测真实坐标](https://blog.csdn.net/qq_43703185/article/details/107225374)

#### 3.2.2 SSD
* 特点：
    * SSD采用VGG16作为基础模型，然后在VGG16的基础上新增了卷积层来获得更多的特征图以用于检测
    * 将VGG16的FC6和FC7层转化为卷积层，如图1上的Conv6和Conv7；去掉所有的Dropout层和FC8层；添加了Atrous算法（hole算法）；Pool5从2x2-S2变换到3x3-S1；抽取Conv4_3、Conv7、Conv8_2、Conv9_2、Conv10_2、Conv11_2层的feature map，然后分别在这些feature map层上面的每一个点构造6个不同尺度大小的bbox，然后分别进行检测和分类，生成多个bbox.[1]
    * 相比VGG，分别将VGG16的全连接层fc6和fc7转换成 3x3卷积层 conv6和1x1卷积层conv7，同时将池化层pool5由原来的stride=2的2x2变成stride=1的3x3，为了配合这种变化，采用了一种Atrous Algorithm，其实就是conv6采用扩展卷积或带孔卷积（Dilation Conv），其在不增加参数与模型复杂度的条件下指数级扩大卷积的视野，其使用扩张率(dilation rate)参数，来表示扩张的大小。普通的3x3卷积，其视野就是3x3，扩张率为2，此时视野变成7x7，扩张率为4时，视野扩大为15x15，但是视野的特征更稀疏了。Conv6采用3x3大小但dilation rate=6的扩展卷积。然后移除dropout层和fc8层，并新增一系列卷积层.[2]
    * 相比yolo1，SSD采用CNN来直接进行检测，而不是像Yolo那样在全连接层之后做检测(对于形状为m x n x p 的特征图，只需要采用3 x 3 x p这样比较小的卷积核得到检测值。)
    * SSD提取了不同尺度的多个特征图来做检测，大尺度特征图（较靠前的特征图）可以用来检测小物体，而小尺度特征图（较靠后的特征图）用来检测大物体
    * 采用了不同尺度和长宽比的先验框（Prior boxes, Default boxes，在Faster R-CNN中叫做锚，Anchors）。在Yolo中，每个单元预测多个边界框，但是其都是相对这个单元本身（正方块），但是真实目标的形状是多变的，Yolo需要在训练过程中自适应目标的形状。而SSD借鉴了Faster R-CNN中anchor的理念，每个单元设置尺度或者长宽比不同的先验框，预测的边界框（bounding boxes）是以这些先验框为基准的，在一定程度上减少训练难度。一般情况下，每个单元会设置多个先验框，其尺度和长宽比存在差异
    * 值得注意的是SSD将背景也当做了一个特殊的类别，如果检测目标共有c 个类别，SSD其实需要预测c+1个置信度值，其中第一个置信度指的是不含目标或者属于背景的评分。后面当我们说c个类别置信度时，请记住里面包含背景那个特殊的类别，即真实的检测类别只有c-1个。
* 基于VGG具体改动:
    * VGG16基本结构见[3.1.2]
    * 修改VGG结构，然后提取了6个特征图：
        1. VGG16的第三个block输出(224对应28，SSD300输入对应输出就是38x38x512)
        2. 把最后一个pool5从2X2,S=2改为3x3,s=1，为了配合这种变化，采用了一种Atrous Algorithm，其实就是conv6采用扩展卷积或带孔卷积（Dilation Conv），其在不增加参数与模型复杂度的条件下指数级扩大卷积的视野。 然后去掉1000的全连接，两个4096的分别改为conv(3x3x1024)和conv(1x1x1024),对应输出是19x19x1024
        3. 在2之后添加两个卷积层conv(1x1x256)、conv(3x3x512),对应输出是10x10x512
        4. 在3之后添加两个卷积层conv(1x1x128)、conv(3x3x256),对应输出是5x5x256
        5. 在4之后添加两个卷积层conv(1x1x128)、conv(3x3x256),对应输出是3x3x256
        6. 在5之后添加两个卷积层conv(1x1x128)、conv(3x3x256),对应输出是1x1x256
    * 其中VGG16中的Conv4_3层将作为用于检测的第一个特征图。conv4_3层特征图大小是38x38 ，但是该层比较靠前，其norm较大，所以在其后面增加了一个L2 Normalization层，以保证和后面的检测层差异不是很大，这个和Batch Normalization层不太一样，其仅仅是对每个像素点在channle维度做归一化，而Batch Normalization层是在[batch_size, width, height]三个维度上做归一化。归一化后一般设置一个可训练的放缩变量gamma。
    * 每个特征图先验框不同，最后一共可以预测8732个边界框。38x38x4+19x19x6+10x10x6+5x5x6+3x3x4+1x1x4=8732，这是一个相当庞大的数字，所以说SSD本质上是密集采样。
* 训练过程
    * 先验框匹配：在训练过程中，首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。在Yolo中，ground truth的中心落在哪个单元格，该单元格中与其IOU最大的边界框负责预测它。但是在SSD中却完全不一样，SSD的先验框与ground truth的匹配原则主要有两点。首先，对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。通常称与ground truth匹配的先验框为正样本（其实应该是先验框对应的预测box，不过由于是一一对应的就这样称呼了），反之，若一个先验框没有与任何ground truth进行匹配，那么该先验框只能与背景匹配，就是负样本。一个图片中ground truth是非常少的， 而先验框却很多，如果仅按第一个原则匹配，很多先验框会是负样本，正负样本极其不平衡，所以需要第二个原则。第二个原则是：对于剩余的未匹配先验框，若某个ground truth的IOU大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配。这意味着某个ground truth可能与多个先验框匹配，这是可以的。但是反过来却不可以，因为一个先验框只能匹配一个ground truth，如果多个ground truth与某个先验框IOU大于阈值，那么先验框只与IOU最大的那个先验框进行匹配。第二个原则一定在第一个原则之后进行，仔细考虑一下这种情况，如果某个ground truth所对应最大IOU小于阈值，并且所匹配的先验框却与另外一个ground truth的IOU 大于阈值，那么该先验框应该匹配谁，答案应该是前者，首先要确保某个ground truth一定有一个先验框与之匹配。[3]
* 损失函数
    * 回归使用Smooth L1：当预测框与 ground truth 差别过大时，梯度值不至于过大；当预测框与 ground truth 差别很小时，梯度值足够小。
    * One-stage目标检测算法需要同时处理定位和识别的任务，即多任务，其损失函数通常是定位损失和分类损失的加权和
    * 然而，在SSD的Caffe源码实现中还有trick，那就是设置variance超参数来调整检测值
* 思考：
    * 为什么要使用空洞卷积：把最后一个pool5从2X2,S=2改为3x3,s=1，参考vgg，224的conv5输出本来是7x7x512，由于s=2，每个点感受野为2，那么后接全连接每个点感受野就是7x2=14，改成3x3的conv后，且pool的s=2变为1，为了保持14的感受野，使用空洞卷积，膨胀后卷积核尺寸 = 膨胀系数 * (原始卷积核尺寸 - 1) + 1  = 6x(3-1)+1 =13约等于14，同时pad也要变为6，元素间隔 = 膨胀系数 - 1 = 6-1 = 5
* 参考：
    * [1] [目标检测之SSD](https://blog.csdn.net/thisiszdy/article/details/89576389)
    * [2] [目标检测|SSD原理与实现](https://zhuanlan.zhihu.com/p/33544892)
    * [3] [SSD算法思想和结构详解](https://www.cnblogs.com/cecilia-2019/p/11342791.html)


#### 3.2.3 Yolo
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


#### 3.2.4 EfficientDet

* 主要贡献点是 BiFPN，Compound Scaling 两部分
* BiFPN: 
    * PANet 只有从底向上连，自顶向下两条路径，作者认为这种连法可以作为一个基础层，重复多次。这样就得到了下图的结果（看中间的 BiFPN Layer 部分）。如何确定重复几次呢，这是一个速度和精度之间的权衡，会在下面的Compound Scaling 部分介绍
    * Cross-Scale Connections: 作者提出之前从FPN 开始普遍采用的，一个特征先 Resize ，再和另一层的特征相加的方式不合理。因为这样假设这两层的特征有了相同的权重。从更复杂的建模角度出发，应该每一个 feature 在相加的时候都要乘一个自己的权重。这样 weighted 的方式能涨 0.4
    * weighted 的时候，权重理论上要用 softmax 归一化到和为1，但由于 softmax 的指数运算开销比较大，作者简化为一个快速的方式 (Fast normalized fusion)，其实就是去掉了 softmax 的指数运算，在 GPU 上能快 30%，性能微微掉一点
    * 总结一下 BiFPN部分，是在PANet的基础上，根据一些主观的假设，做了针对性的化简，得到了参数量更少，效果更好的连接方式。
* Compound Scaling:
    * EfficientNet 在 Model Scaling 的时候考虑了网络的 width, depth, and resolution 三要素。而 EfficientDet 进一步扩展，把 EfficientNet 拿来做 backbone，这样从 EfficientNet B0 ~ B6，就可以控制 Backbone 的规模；neck 部分，BiFPN 的 channel 数量、重复的 layer 数量也可以控制；此外还有 head 部分的层数，以及 输入图片的分辨率，这些组成了 EfficientDet 的 scaling config 。

* 参考：
    * [1] [EfficientDet 算法解读](https://zhuanlan.zhihu.com/p/93241232)


### 3.3 分割
#### 3.3.1 U-net


### 3.4 识别
#### 3.4.1 FaceNet
* CNN提取特征，L2归一化到128维向量，然后通过三元组损失优化（没有使用softmax），使用这种方式学到的图像表示非常紧致，使用128位足矣。元组的选择非常重要，选的好可以很快的收敛。
* Triplet Loss：三元组就是三个样例，如(anchor, pos, neg)，其中，x和p是同一类，x和n是不同类。那么学习的过程就是学到一种表示，对于尽可能多的三元组，使得anchor和pos的距离，小于anchor和neg的距离
* 参考:
    * [1] [FaceNet--Google的人脸识别](https://blog.csdn.net/stdcoutzyx/article/details/46687471)

## 四、优化、实现

#### 4.1 im2col
* 卷积一般通过一种im2col方式实现。熟悉卷积都知道，感受野区域是个二维的区域，这二维的区域从存储上不是连续的，这样就不便于计算。im2col算法完成的就是通过矩阵乘法来完成卷积核和感受野的对应相乘，这样计算起来就方便多了。
* 实现：
    * 单通道图像，单通道卷积核为例
    * 图像：卷积核对应的一个取样区域被转换为一个行向量row；从左到右从上到下有M个取样区域，则图像转化为[M, len(row)]大小的矩阵；
    * 卷积核：单通道的卷积核被拉伸为一个列向量col，大小为[len(col), 1]；len(row)=len(col)；
    * 输出：[M, 1]
* 参考:
    * [1] [卷积详解之im2col算法](https://august-us.blog.csdn.net/article/details/104709433)
    * [2] [im2col卷积运算实现](https://zhuanlan.zhihu.com/p/297796588)

#### 4.2 wingrad卷积

## 五、应用

### 4.1 OCR
#### 4.1.1 CRNN

#### 4.1.2 CPTN

## 六、其他

### 6.1 网络架构

#### 6.1.1 Siamese Network
* 一般用于比较两个输入是否近似
* 参考:
    * [1] [孪生网络（Siamese Network）](https://blog.csdn.net/weixin_45657358/article/details/108707358)
    * [2] [【深度学习】Siamese Network](https://blog.csdn.net/qq_34106574/article/details/84028665)
    * [3] [Facial Similarity with Siamese Networks in Pytorch](https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch)

#### 6.1.2 Triplet Network
* 基于6.1.1的改进
* 思考：
    * Triplet Network 、Triplet Loss
        1. 最早提出triplet loss的想法的是2003年的Learning a distance metric from relative comparisons
        2. 14年的论文Deep metric learning using Triplet network提出了Triplet network的概念，是针对Siamese Networ改进而来的架构，因此得名
        3. 15年Facenet论文提出，里面讲到了triplet loss，用于自己的一个深度CNN网络。
        4. 总结：triple loss可以应用于很多网络、架构和任务，可以理解为一个类似于交叉熵的通用loss，triplet network是单独特指相对于Siamese Network的一种网络架构，只是这种网络架构的loss可以用triple loss也可以不用，比如用MarginRankingLoss。所以网上大部分文章混为一谈了，且正文都是在讲triplet loss。

* 参考:
    * [1] [论文笔记：Triplet Network](https://blog.csdn.net/hongbin_xu/article/details/83064290)
    * [2] [Wiki:Triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)

#### Seq2Seq
