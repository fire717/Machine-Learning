"""
@Fire
focal loss本身是用于检测网络的
其中alpha因子用于控制不平衡样本比例，主要是降低背景类
gamma因子用于控制易分类样本权重

而在分类任务中，没有背景类，且keras的fit可以直接设置class_weight，所以这里直接去掉了alpha

"""
# version 1
def focal_loss(y_true,y_pred,gamma = 2):
    '''
    :param y_true: ont-hot encoding ,shape is [batch_size,nums_classes]
    :param y_pred: shape is [batch_size,nums_classes],each example defined as probability for per class
    :return:shape is [batch_size,], a list include cross_entropy for per example
    '''
    y_pred = K.clip(y_pred, K.epsilon(),1.0 - K.epsilon())
    crossEntropyLoss = -((1-y_pred)**gamma)*y_true * tf.log(y_pred)#facal loss
 
    return tf.reduce_sum(crossEntropyLoss,-1)

# version 2 rec
def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)
