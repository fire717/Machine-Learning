# 手写汉字识别调研
> @Yiwei 2019-05

## 1. 论文
### 1.1 直接针对离线中文汉字手写识别的论文
#### [2019 CVPR] Handwriting Recognition in Low-resource Scripts using Adversarial Learning
代码(但目前还没有更新):https://github.com/AyanKumarBhunia/Handwriting_Recogition_using_Adversarial_Learning

针对整行手写拉丁文识别。在手写识别中，现有的预处理策略很难使得训练数据包含所有可能的变化，该文提出了一个对抗特征变形模块（AFDM），学习如何通过可扩展的方式对特征进行弹性扭曲变化。AFDM被插入到中间层之间，并与原始框架交替训练，使其更好地学习高层信息特性而不是琐碎细节特征。

论文介绍了在英文手写图像信息检索中的两类方法：

1. 手写单词识别（HWR）。即直接对文字图像进行识别。
2. 手写单词定位（HWS）。用于从示例单词图像集合中查找所要查询关键字（字符串或示例字图像）的出现，单词定位（word spotting, 参考文献[Word Spotting for Historical Documents](http://ciir.cs.umass.edu/pubfiles/mm-409.pdf) ）指的是把整个文档分割成单个的单词图像，然后使用图像匹配来计算单词图像之间的成对“距离”，它可以用来对手写文档集合中出现的所有单词进行分类聚类。

论文在两类方法上分别选取了baseline（CRNN 和 Phocnet）加入了AFDM模块，都取得了提升。

论文分析相关文献时提到，目前大多数的新研究中，最重要的几乎都是数据预处理部分。分别查看了对应论文，数据处理方法大致如下：

1. Best practices for convolutional neural networks applied to visual document analysis. ICDAR 2003 ：通过仿射变换来实现平移、旋转、扭曲。加入弹性变形效果提升更好，通过随机位移场来实现，详情参考论文。
2. Imagenet classification with deep convolutional neural networks. 2012 ：它用到的一种增强方法是改变训练图片的RGB的强度。 对图片中每一个RGB像素点进行一种变换，通过类似PCA的方法，求出特征向量和特征值，来提取主元。但是只对彩图多类物体有效。
3. Cnn-n-gram for handwriting word recognition. CVPR 2016 ： 原始图像为32*100的灰度图，随机中心旋转{−5, −3, −1, +1, +3, +5}度，剪切变换{−0.5, −0.3, −0.1, 0.1, 0.3, 0.5}，共6*6=36种变换，同时对测试集做了同样的变换。

论文介绍了AFDM的基本构成以及引入AFDM具体位置是在CRNN的conv4_1层后面，但是其具体实现论文中貌似没有提到，github上也没有代码。

#### [2018] Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks
在ICFHR2018自动文本识别竞赛中获胜。论文提出了一种简单新颖的通用识别模型，使用的纯CNN+CTC。同样也提出了一系列的通用数据增强方法。

论文首先对相关文献作了一个综述，指出目前给予深度学习的序列转换有两大主流：

1. 基于CTC的模型：一些之前SOTA的方案都是基于CNN+RNN+CTC，而最近都慢慢转向CNN+CTC，不仅并行能力和速度有了提升，重新设计的CNN结构使得准确率也达到了新的SOTA。
2. encoder-decoder模型：一个CNN+BLSTM做encoder，LSTM做解码的方案取得了不错的效果，但是其预测时使用beam-search耗时非常高。

论文提出模型的一些细节：

1. 提出了一种叫做GateBlocks的结构；同时用depthwise separable convolution代替一般的CNN，相比之下可以取得同样或者更好的分类表现，同时参数量极大的减少速度也更快。
3. 在depthwise convolution 和1x1 pointwise convolution之间使用Batch Normalization 很重要。 
4. 使用spatial DropOut比一般的un-structured DropOut提供更好的泛化能力.（普通的dropout会随机地将部分元素置零，而SpatialDropout会随机地将部分区域置零）

同时也讲了几种针对手写字体的数据增强方案：

1. 投影变换（透视变换/Projective Transformations）。论文发现或者改变所有四个点的x坐标，固定y坐标，或者反过来，无论是垂直还是水平地改变图像，但不同时改变两者，几乎总是更好的；在垂直或水平改变时，任何连接的斜对角之间的新距离不得小于原始的一半，或大于原始的两倍。
2. 弹性变形。相比应用到每个像素，而是生成网格对每个网格应用。论文同样将变形应用于X方向或Y方向，而不是两者。也发现在这里一次对一个维度应用扭曲是更好的。同时强制变形网格矩形的最小宽度或高度为1。换句话说，尽管可能，但不允许为零或负值。
3. 符号翻转。由于文本内容对其颜色或纹理是不变的，而最简单的方法之一就是对输入向量进行符号翻转来强调这种不变性。必须强调的是，即使所有的训练和测试图像都具有相同的文本和背景色，这种增强也会提高性能。其实应该就是颜色反转。

#### [2019] A Scalable Handwritten Text Recognition System
针对英文手写识别。相比一般关注提升准确率的论文，本论文关注手写汉字在线数据的有效收集以及有效的集成到系统中，同时本文使用的识别模型也没有使用LSTM，且和使用了LSTM的模型能达到一样的效果，但是并行性、效率等方面更好。

#### [2019] End to End Recognition System for Recognizing Offline Unconstrained Vietnamese Handwriting
针对手写越南字体识别。用了encoder-decoder结构，encoder是densenet用于提取图像特征，decoder是加入了attention的LSTM。同样因为越南字体也属于拉丁文系，是由单个字母组成单词，所以LSTM及attention是有意义的，针对汉字意义不大。

#### [2018] A High-Performance CNN Method for Offline Handwritten Chinese Character Recognition and Visualization
针对单个手写汉字识别。提出了一个全局加权平均池化的瓶颈层。

#### [2018] Deep Template Matching for Offline Handwritten Chinese Character Recognition
对单个手写汉字和标签进行模版匹配对比，判断是否是同一个字。看做了一个二分类问题。这个做手写签名比对的话可以参考。

#### [2018] Writer-Aware CNN for Parsimonious HMM-Based Offline Handwritten Chinese Text Recognition
针对单个手写汉字识别，将混合卷积神经网络和隐马尔可夫模型（CNN-HMM）引入到离线手写体中文文本识别中。

#### [2015] High Performance Offline Handwritten Chinese Character Recognition Using GoogLeNetand Directional Feature Maps
代码:https://github.com/zhongzhuoyao/HCCR-GoogLeNet

针对单个手写汉字识别，提出的HCCR-Googlenet有19层深，但只有726万个参数。使用了八个方向的梯度特征和八个方向腐蚀图梯度特征以及HOG特征。

#### [2014] Dropout improves Recurrent Neural Networks for Handwriting Recognition

当时手写识别效果最好的方案还是MDLSTM，论文提出一种对RNN加入dropout的方案可以有效提升准确率。

### 1.2 其它相关论文
#### [2016] Robust Scene Text Recognition with Automatic Rectification
RARE模型通过一个空间变换网络来矫正，在识别变形的图像文本时效果很好。具体识别模型为CNN+BLSTM+ATT_GRU+Softmax。

#### [2016] Scan, Attend and Read: End-to-End Handwritten Paragraph Recognition with MDLSTM Attention
提出了一种基于注意力机制的，端到端的整段识别多行手写文字（英文）的方法。如果要做多行汉字可以参考。

#### [2015] DeepFont: Identify Your Font from An Image
英文字体风格识别。如果做汉字风格识别可以参考。

#### [2014] Reading Text in the Wild with Convolutional Neural Networks
自然场景下的英文识别，直接识别单词而不是字母。把文字的识别转变为分类问题。只不过分类的类别数非常非常多，是一个个的单词，总共有 90K个单词。 汉字识别本质也是分类。


## 2. Github项目
开源的专门针对中文手写汉字识别方案很少，都是针对单个汉字的。比如：

* [DeepHCCR](https://github.com/chongyangtao/DeepHCCR) 基于GoogLeNet的单个手写汉字识别。

实验观察目前的通用中文OCR的体征提取能力比较强，因此在数据量足够大的情况下，使用通用中文OCR识别也能取得一定效果。而针对通用的中文OCR，目前都是整行识别的方案，比较主流的有：

* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 谷歌的OCR识别开源项目，最早是基于传统的方法，先分割再识别，对中文效果不好。2018年8月发布的4.0.0版本加入了LSTM，效果未测试。
* [chinese_ocr](https://github.com/YCG09/chinese_ocr) 使用DenseNet + CTC整行识别，在验证集上整行准确率0.983。
* [chineseocr](https://github.com/chineseocr/chineseocr) 使用CRNN（其CNN是原始CRNN论文中的7层网络）整行识别，效果还可以。
* [CHINESE-OCR](https://github.com/xiaofengShi/CHINESE-OCR) 使用CRNN(CNN+GRU/LSTM+CTC)整行识别，和chineseocr类似，效果还可以。
* [Attention-ocr-Chinese-Version](https://github.com/A-bone1/Attention-ocr-Chinese-Version) 使用了CNN+RNN+Atention验证集单字准确率0.929，整行准确率0.802.
* [caffe_ocr](https://github.com/senlinuc/caffe_ocr) 主流ocr算法研究实验性的项目，目前实现了CNN+BLSTM+CTC架构.其实验结果最好的一个是densenet-no-blstm-vertical-feature，即CNN使用densenet，不使用RNN，最后一层去掉1*4的pooling保留更多垂直特征（chinese_ocr就是用的这个方案）。在验证集准确率0.982.



## 3.数据集
* [CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) 中科院自动化研究所开源的手写汉字数据集，提供了在线手写数据和离线手写数据，包含中文汉字和中英文字符共计7365个，目前我们使用的是常用汉字3755个，每一个字约有200多个不同的手写图片。


## 4.总结
在纯CNN（densenet）能达到效果的前提下，个人觉得对于通用识别模型来说，引入RNN或者是attention是没有必要也没有意义的，因为它们都是关注序列上下文特征，而对于通用识别模型来说，不应该有特定的上下文，且很多论文用RNN或者attention来做手写字体识别都是针对拉丁文，即为字母组成的单词，这种就存在序列信息，因为一个单词你错了一个字母就是错的，而对于汉字来说，单个字就是一个完整的整体了，通过构造样本测试腾讯的通用手写识别模型应该也是没有利用序列信息的。

同理，对于通用模型，引入NLP技术也是没有必要的。但是如果后续做特定场景手写体识别，加入NLP技术有两种可参考的思路：

* 直接用一个RNN网络，输入是最后解码后输出的结果序列，输出是整行标签。也可以用seq2seq或者transformer来做。
* 对于输出序列的每一个字，取top5/top10，然后通过n元模型来选择所有组合句子中整体置信度最高的作为最终输出。

对于目前的基于densenet的手写汉字识别的提升方向，在已有模型能够对训练数据达到一定拟合能力的情况下，核心是提升泛化能力，主要有两个可行的方向：

1. 模型层面：

	* 加入不同层数以及不同dropout比例的dropout层；
	* 尝试不同的结构（例如论文中第二篇）；
	
2. 数据层面：

	* 收集更多真实手写数据用于训练和固定的测试；
	* 造更多的数据，光数据量级增加不行，还要变化性足够；
	* 尝试更多的数据增强方案，目前已经增加了弹性变形，可以考虑增加些随机遮挡、颜色反转等（也可参考论文中第二篇）。
