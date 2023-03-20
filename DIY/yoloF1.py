import os
import numpy as np


def IOU(Reframe,GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
    """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]

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


def file2box(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        items = line.strip().split(' ')
        boxes.append([float(x) for x in items])

    return boxes


def box2res(gt_boxes, pre_boxes, score_th,class_num,iou_th=0.5):
    tp_list = [0 for _ in range(class_num)]
    fp_list = [0 for _ in range(class_num)]
    fn_list = [0 for _ in range(class_num)]
    
    pre_boxes = [box for box in pre_boxes if box[-1]>score_th]

    for idx in range(class_num):
        gt_boxes_cls = []
        for box in gt_boxes:
            if box[0]==idx:
                gt_boxes_cls.append(box)
        pre_boxes_cls = []
        for box in pre_boxes:
            if box[0]==idx:
                pre_boxes_cls.append(box)
        #print("gt_boxes_cls: ",gt_boxes_cls)
        #print("pre_boxes_cls: ",pre_boxes_cls)
        tp,fp,fn = 0,0,0
        if len(gt_boxes_cls)==0:
            fp = len(pre_boxes_cls)
        elif len(pre_boxes_cls)==0:
            fn = len(gt_boxes_cls)
        else:
            for gt_box in gt_boxes_cls:
                matched = False
                for pre_box in pre_boxes_cls:
                    if pre_box[-1]<0: #pass matched
                        continue

                    iou_score = IOU(gt_box[1:5], pre_box[1:5])
                    if iou_score>iou_th:
                        matched = True
                        pre_box[-1] = -1
                        break
                
                if matched:
                    tp+=1
                else:
                    fn+=1
            fp = len([box for box in pre_boxes_cls if box[-1]>0])
        #print("tp,fp,fn: ", tp,fp,fn)
        tp_list[idx] += tp
        fp_list[idx] += fp
        fn_list[idx] += fn





    return tp_list,fp_list,fn_list




score_th = 0.25
class_name = [ 'open', 'close']
class_num = len(class_name)

label_dir = "../data/test/labels"
pre_dir = "runs/detect/exp62/labels"


label_names = os.listdir(label_dir)
pre_names = os.listdir(pre_dir)

tp_all_list = np.array([0 for _ in range(class_num)])
fp_all_list = np.array([0 for _ in range(class_num)])
fn_all_list = np.array([0 for _ in range(class_num)])
for i,label_name in enumerate(label_names):
    gt_boxes = file2box(os.path.join(label_dir, label_name))

    if label_name in pre_names:
        pre_boxes = file2box(os.path.join(pre_dir, label_name))
    else:   
        pre_boxes = []
    #print("gt_boxes: ",gt_boxes)
    #print("pre_boxes: ",pre_boxes)
    tp_list,fp_list,fn_list = box2res(gt_boxes, pre_boxes, score_th,class_num)
    #print("tp_list,fp_list,fn_list: ",tp_list,fp_list,fn_list)
    tp_all_list+=np.array(tp_list)
    fp_all_list+=np.array(fp_list)
    fn_all_list+=np.array(fn_list)

#print(tp_all_list,fp_all_list,fn_all_list)
#cal F1
p_list = []
r_list = []
f1_list = []
print(("----------------------------------"))
for idx in range(class_num):
    p = tp_all_list[idx]/(tp_all_list[idx]+fp_all_list[idx]+1e-7)
    r = tp_all_list[idx]/(tp_all_list[idx]+fn_all_list[idx]+1e-7)
    f1 = 2*p*r/(p+r+1e-7)
    p_list.append(p)
    r_list.append(r)
    f1_list.append(f1)
    print("%s: P:%.4f R:%.4f F1:%.4f" % (class_name[idx],p,r,f1))

print("Mean: P:%.4f R:%.4f F1:%.4f" % (np.mean(p_list),np.mean(r_list),np.mean(f1_list)))
print(("----------------------------------"))


