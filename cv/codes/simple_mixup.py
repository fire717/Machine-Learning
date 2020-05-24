import numpy as np
import cv2

"""
说明：
这里是前期看到公式后一个粗略的实现
参考了更多资料后其实有些地方还是有点问题

比如weight应该是一个beta分布而不是均匀分布，如
weight = np.random.beta(alpha,alpha)
    
然后是输入每个batch的x，y即可，统一和一个从数据集中随机选取的图片做mixup

最后，最关键的是，最后计算loss并不是修改标签y，
而是如下计算loss：
loss = weight * criterion(outputs, targets_a) + (1 - weight) * criterion(outputs, targets_b)；

下一次用到mixup的时候修改了再更新到这里吧。
5.24： 已更新beta分布
"""

def mixup_batch(x1,y1,x2,y2,alpha=0.4):
    """
    get batch data
    :param x: two training imgs (same shape)   ndarry
    :param y: two one-hot labels(same shape)   ndarry
    :param alpha: hyper-parameter α, default as 0.2
    :return: new_x,new_y
    """
    #weight = np.random.choice([0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9])
    weight = np.random.beta(alpha, alpha)
    print(weight)
    new_x = x1*weight+x2*(1-weight)
    new_y = y1*weight+y2*(1-weight)

    return new_x, new_y


img1 = cv2.imread("1.jpg")
img1 = cv2.resize(img1,(224,224))
img2 = cv2.imread("2.jpg")
img2 = cv2.resize(img2,(224,224))
y1 = np.array([0,0,1])
y2 = np.array([0,1,0])

x,y = mixup_batch(img1,y1,img2,y2)
cv2.imwrite("12.jpg", x)
print(y)
