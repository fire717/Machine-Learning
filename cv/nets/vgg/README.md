# VGGNet
> @Fire 2019.7.2

* Intro: 2014年新的一届ILSVRC大赛中Googlenet与VGG的身影分外亮眼。Googlenet相对VGG而言在网络结构上有了更新的突破，不过其复杂度也大大增加了。VGG相对Googlenet虽然精度略逊些，但其整体网络框架还是延续了Alexnet及更早的Lenet等的一贯思路，此外还更深入的探讨了ConvNet深度对模型性能可能的影响。

* Year: 2014
* Paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](http://xueshu.baidu.com/usercenter/paper/show?paperid=2801f41808e377a1897a3887b6758c59&site=xueshu_se)
* Code: [keras_vgg](keras_vgg.py)
* Info: 224 * 224 * 3的输入，10的输出，参数量为1.3亿。

![net](./vgg.jpg)

![net](./vgg16.jpg)

* Note: 

	1. 相比于AlexNet最大的改进是用小size的Filter代替大size的Filter。两个3 * 3的卷积核代替5 * 5的卷积核，三个3 * 3代替7 * 7。多个小尺度的卷积核比大尺度的卷积核非线性更强，同时参数减少，不容易过拟合。

