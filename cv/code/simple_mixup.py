import numpy as np
import cv2

def mixup_batch(x1,y1,x2,y2):
    """
    get batch data
    :param x: two training imgs (same shape)   ndarry
    :param y: two one-hot labels(same shape)   ndarry
    :param alpha: hyper-parameter α, default as 0.2
    :return: new_x,new_y
    """
    weight = np.random.choice([0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9])
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
