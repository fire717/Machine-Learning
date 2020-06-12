# Machine-Learning

[![Travis](https://img.shields.io/travis/rust-lang/rust.svg)](https://github.com/fire717/Machine-Learning) [![PyPI](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/fire717/Machine-Learning) [![Chrome Web Store](https://img.shields.io/chrome-web-store/price/nimelepbpejjlbmoobocpfnjhihnpked.svg)](https://github.com/fire717/Machine-Learning) [![Chrome Web Store](https://img.shields.io/chrome-web-store/stars/nimelepbpejjlbmoobocpfnjhihnpked.svg)](https://github.com/fire717/Machine-Learning)


## 0.不调库系列 
> No free lunch.

* [线性回归](/DIY/ex1_py_liner.ipynb) - [logistic回归](/DIY/LR.ipynb) - [感知机](/DIY/perceptron.ipynb) - [SVM(SMO)](/DIY/SVM.ipynb) - [神经网络](/DIY/NN.ipynb) 
* [决策树](/DIY/DecisionTree.ipynb) - [Adaboost](/DIY/Adaboost.ipynb) 
* [kNN](/DIY/kNN.ipynb) - [朴素贝叶斯](/DIY/NaiveBayes.ipynb)
* [EM](/DIY/EM.ipynb) - [HMM](/DIY/HMM.ipynb) - [条件随机场](/DIY/CRF.ipynb)
* [kMeans](/DIY/kMeans.ipynb) - [PCA](/DIY/PCA.ipynb)
* [ROC曲线&AUC值](/DIY/ROC_AUC.ipynb)
* [Stacking](./DIY/Stacking.py)（[demo](/DIY/tryStacking.ipynb)）
* [计算IOU](./DIY/IOU.py)

> 参考：《机器学习》周志华  《统计学习方法》李航


## 1.机器学习&深度学习

  工具   |     书籍      |    课程     |    比赛 |   框架
---------|---------------|-------------|-------- |---------
 [Scikit-learn](/Base/tools/scikit-learn)| [机器学习实战](/Base/books/ML_in_action)  | [机器学习/Andrew Ng](/Base/courses/coursera_ML)      | [Kaggle](/Base/challenge/kaggle) | [Keras ★](/Base/frameworks/keras)
  [XGBoost](/Base/tools/xgboost)  | [集体智慧编程](/Base/books/JTZHBC)     | [Deep Learning/Andrew Ng](/Base/courses/DL_AndrewNg) | [天池](/Base/challenge/tianchi) | [Tensorflow](/Base/frameworks/tensorflow)
 [LightGBM](/Base/tools/lightgbm) |      |      | [Biendata](/Base/challenge/biendata) |  [PyTorch](/Base/frameworks/pytorch)
[Spark](/Base/tools/spark)|  |  | [AI challenge](/Base/challenge/AIchallenge) |[Mxnet](/Base/frameworks/mxnet)
|  |  |  |  [DataFountain](/Base/challenge/DataFountain)  | [Caffe](/Base/frameworks/caffe)


## 2.自然语言处理
* 课程：[自然语言处理班 - 七月在线](/Base/courses/qiyuezaixian) | CS224n
* 工具：[NLTK](/NLP/tools/nltk) | [jieba](/NLP/tools/jieba) | [gensim](/NLP/tools/gensim) | [NLPIR](/NLP/tools/NLPIR) | [word2vec](/NLP/tools/word2vec) | [LDA](./NLP/tools/lda) | BLEU([介绍](https://blog.csdn.net/qq_31584157/article/details/77709454)、[使用](https://cloud.tencent.com/developer/article/1042161))

* 代码：[社交网络语言re](/NLP/codes/re.ipynb) |
* 开源代码: [funNLP](https://github.com/fighting41love/funNLP) | [AI写诗](https://github.com/jinfagang/tensorflow_poems) | 


## 3.计算机视觉
#### 常用代码
* 常用预处理：[打乱数据集](https://github.com/fire717/Python-Learner/blob/master/tools/numpy/tools/transformation_data.py) | [计算图像均值方差](https://github.com/fire717/Python-Learner/blob/master/tools/numpy/tools/compute_imgs_mean_std.py) |  [分类数据增强](https://github.com/fire717/Python-Learner/blob/master/tools/OpenCV/dataAugmentation_classify.py) |  [检测数据增强](https://github.com/fire717/Python-Learner/blob/master/tools/OpenCV/dataAugmentation_objectdetect.py) | 

* VOC数据集相关：[生成目录结构](./cv/codes/makeVOCDirs.py) | [标签转xml写入](./cv/codes/flickr_to_voc.py) | [可视化标签框](./cv/codes/show_voc_box.py) | [更新训练验证txt图片名](./cv/codes/updateTXT.py) )| [VOC转csv格式](./cv/codes/pascalVOC2csv.py) | [计算csv格式数据合适的anchor](https://github.com/martinzlocha/anchor-optimization/)

* 常用算法：[NMS](/cv/codes/nms.py) | [Mixup](/cv/codes/simple_mixup.py) | [label_smoothing](/cv/codes/label_smoothing.py) | [Weighted-Boxes-Fusion(NMS,WBF..)](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
#### 其他
* 课程: [CS231n/Feifei Li](/Base/courses/cs231n) 
* 笔记: [chinese-ocr项目](/CV/note/chineseocr-ctpn-densenet.md) | [《深度卷积网络：原理与实践》读书笔记](./CV/note/DCNN_book_note.md) | [手写汉字识别调研](/CV/note/handwrite_ocr_note.md)
* 经典分类网络： [LeNet-5](/CV/nets/lenet5/) | [AlexNet](/CV/nets/alexnet/) | [VGGNet](/CV/nets/vgg/) | GoogleNet | ResNet | DenseNet | Xception | [EfficientNet](https://github.com/titu1994/keras-efficientnets)
* 经典检测网络： [SSD](https://github.com/pierluigiferrari/ssd_keras) | FasterRCNN | Yolo | CornerNet | CenterNet | [EfficientDet](https://github.com/xuannianz/EfficientDet)
* 经典分割网络：Unet | DeepLab
* 经典项目(已验证跑通)： [HyperLPR车牌识别](https://github.com/zeusees/HyperLPR) | [中文OCR1（YOLOv3+CRNN）](https://github.com/chineseocr/chineseocr)、 [中文OCR2（CTPN + DenseNet）](https://github.com/YCG09/chinese_ocr) | [RFBNet(ECCV2018快速目标检测)](https://github.com/ruinmessi/RFBNet) | [AlphaPose(人体姿态估计)](https://github.com/MVIG-SJTU/AlphaPose) | [libface快速人脸检测](https://github.com/ShiqiYu/libfacedetection) | [轻量级性别年龄分类模型](https://github.com/deepinsight/insightface/tree/master/gender-age) | [图像风格迁移](https://github.com/wenqiwenqi1/neural-style) | [超分辨率](https://github.com/titu1994/Image-Super-Resolution) | [人脸比对DeepID](https://github.com/shen1994/DeepID) | [Mask_RCNN人体关键点、分割](https://github.com/Superlee506/Mask_RCNN_Humanpose) | [人像卡通化](https://github.com/minivision-ai/photo2cartoon)


## 4.数据挖掘
* 笔记：[特征工程](/Others/note/FeatureEngneering.md)

## 5.其他
* [资源索引](/Others/infos)
* [★知识点整理](/Others/mlthings.md)




