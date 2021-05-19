#参考https://blog.csdn.net/weixin_38241876/article/details/110041645
#但是它计算有问题，这里是修改后的

def myIOULoss(self,predicted_locations, gt_locations, labels,
                GIoU=False, DIoU=False, CIoU=False):
    #torch.Size([691, 4]) torch.Size([691, 4])

    #    
    ### 1. to conner type box
    pos_mask = labels > 0
    pre_boxes = box_utils.convert_locations_to_boxes(
                        predicted_locations, self.priors, 0.1, 0.2)
    pre_boxes = box_utils.center_form_to_corner_form(pre_boxes)
    pre_boxes = pre_boxes[pos_mask, :].reshape(-1, 4)
    #print(pre_boxes[:5])#[0.3799, 0.2177, 0.4424, 0.2723]

    gt_boxes = box_utils.convert_locations_to_boxes(
                        gt_locations, self.priors, 0.1, 0.2)
    gt_boxes = box_utils.center_form_to_corner_form(gt_boxes)
    gt_boxes = gt_boxes[pos_mask, :].reshape(-1, 4)
    #print(gt_boxes[:5])
    #print(pre_boxes.shape, gt_boxes.shape)
    num_pos = gt_boxes.size(0)


    ### 2.compute IOU
    b1_x1, b1_y1, b1_x2, b1_y2 = pre_boxes[:,0], pre_boxes[:,1], pre_boxes[:,2], pre_boxes[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    #print(inter.shape, union.shape)
    iou = inter / union  # iou
    # print(iou.shape) #[691]
    # b
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            loss = iou - (c_area - union) / c_area  # GIoU
            loss = 1-loss
        else:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                loss = iou - rho2 / c2  # DIoU
                loss = 1-loss
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                loss = iou - (rho2 / c2 + v * alpha)  # CIoU
                loss = 1-loss

        loss = loss.sum()
    else:
        iou = -torch.log(iou + 1e-16) #防止为0
        loss = iou.sum()

    #print(loss,num_pos)
    # b
    return loss, num_pos
