import numpy as np

_IOU_threshold = 0.6

def IOU(Reframe,GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
    """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width*height # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio

def computeLoss(pre_box_list, label_box_list, R_weight = 1):
    pre_box_list = np.array(pre_box_list)
    label_box_list = np.array(label_box_list)
    total_pre = len(pre_box_list)
    total_label = len(label_box_list)

    # compute precise
    p_count = 0.0
    for box_pre in pre_box_list:
        for box_label in label_box_list:
            print(IOU(box_pre,box_label))
            if IOU(box_pre,box_label) > _IOU_threshold:
                p_count += 1
                break
    P = p_count / total_pre

    # compute recall
    r_count = 0.0
    for box_label in label_box_list:
        for box_pre in pre_box_list:
            if IOU(box_pre,box_label) > _IOU_threshold:
                r_count += 1
                break
    R = r_count / total_label

    # compute F1-score

    F = 2*P*R*R_weight/(P+R*R_weight)

    return P,R,F


if __name__ == '__main__':
    
    # 1.test iou
    # tests_iou = [
    #                 [ [[10,40,30,80],[10,40,30,80]], 1],
    #                 [ [[10,40,30,80],[30,80,60,120]], 0]
    #             ]

    # for t in tests_iou:
    #     v,_,_ = IOU(t[0][0],t[0][1])
    #     print(v, t[1])
    
    # 2.test compute
    pre_box_list = [ [10,40,30,80], [30,80,60,120], [40,100,80,140], [42,100,80,140], [44,100,80,140] ]
    label_box_list = [ [10,40,30,80], [30,80,60,120] ]
    print(computeLoss(pre_box_list, label_box_list))
