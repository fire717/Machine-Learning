import numpy as np
def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        print("-----------------")
        print("order[1:]: ",order[1:])
        print("x1[order[1:]]: ",x1[order[1:]])
        print("np.maximum(x1[i], x1[order[1:]]): ", np.maximum(x1[i], x1[order[1:]]))
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def soft_nms(boxes, scores, iou_thres, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= iou_thres
    rule out overlap > iou_thres
    :param dets: [[x1, y1, x2, y2 score]]
    :param iou_thres: retain overlap <= iou_thres
    :return: indexes to keep
    """
    min_score = 0.001

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    #score降序的id号

    keep = []
    while order.size > 0:
        i = order[0]

        if scores[i]<min_score:
            order = order[1:]
            continue

        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        #inds_less = np.where(ovr <= iou_thres)[0]
        inds_large = np.where(ovr > iou_thres)[0]
        #原始只保留小于iou阈值的框
        #order = order[inds_less + 1]
        #改进：保留小于iou阈值的框，对于大于iou阈值的框，修改其score
        # print(len(scores[inds_large + 1]), len(ovr[inds_large]))
        # print(scores[inds_large + 1][:20],ovr[inds_large][:20],1-ovr[inds_large][:20])
        #order[inds_large + 1] = order[inds_large + 1]*(1-ovr[inds_large])
        scores[inds_large + 1] = scores[inds_large + 1]*(1-ovr[inds_large])
        # print(scores[inds_large + 1][:20])
        # b
        order = order[1:]

    return np.array(keep)
