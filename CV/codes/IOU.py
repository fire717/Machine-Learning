def IOU(box1,box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 求交集部分左上角的点
    xmin = max(xmin1,xmin2)
    ymin = max(ymin1,ymin2)
    # 求交集部分右下角的点
    xmax = min(xmax1,xmax2)
    ym=ax = min(ymax1,ymax2)
    # 计算输入的两个矩形的面积
    s1 = (xmax1-xmin1) * (ymax1 - ymin1)
    s2 = (xmax2-xmin2) * (ymax2 - ymin2)

    #计算总面积
    s = s1 + s2 
    # 计算交集
    inter_area = (xmax - xmin) * (ymax - ymin)

    iou = inter_area / (s - inter_area)
    return iou
