# used in my ssd
# from Libra RCNN

def IOU_balanced(predicted_locations, gt_locations, labels, priors, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        predicted_locations (32*4420*4)
        gt_locations (32*4420*4)
        labels (32*4420): the labels.
        priors:anchor
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    # print(num_pos.sum(), num_neg.sum()) # shape 32*1/691 2073

    ### 1. to conner type box
    pre_boxes = convert_locations_to_boxes(
                        predicted_locations, priors, 0.1, 0.2)
    pre_boxes = center_form_to_corner_form(pre_boxes)
    #pre_boxes = pre_boxes[neg_mask, :]#.reshape(-1, 4)

    #print(pre_boxes[:5])#[0.3799, 0.2177, 0.4424, 0.2723]
    gt_boxes = convert_locations_to_boxes(
                        gt_locations, priors, 0.1, 0.2)
    gt_boxes = center_form_to_corner_form(gt_boxes)
    #gt_boxes = gt_boxes[neg_mask, :]#.reshape(-1, 4)


    ### 2.compute IOU
    b1_x1, b1_y1, b1_x2, b1_y2 = pre_boxes[:,:,0], pre_boxes[:,:,1], pre_boxes[:,:,2], pre_boxes[:,:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = gt_boxes[:,:,0], gt_boxes[:,:,1], gt_boxes[:,:,2], gt_boxes[:,:,3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter
    
    #print(inter.shape, union.shape)
    iou = inter / union  # iou  32*4420
    iou[pos_mask] = -1


    ### main
    #default k=3 0-0.5 -> 0.167, 0.332
    sample_every_bin = num_pos * neg_pos_ratio / 3
    # print(sample_every_bin.sum()) #691



    neg_mask_k1 = (iou>=0.332) & (iou<0.5)
    #print(neg_mask_k1.shape, len(neg_mask_k1)) #torch.Size([32, 4420]) 32
    #print(neg_mask_k1[0].shape)
    for i in range(len(neg_mask_k1)):
        idx = torch.where(neg_mask_k1[i]==True)[0]
        #print(len(idx), sample_every_bin[0], sample_every_bin[0].item())
        change_num = len(idx)-int(sample_every_bin[i].item())
        if change_num>0:
            idx_change = idx[torch.randperm(len(idx))[:change_num]]
            neg_mask_k1[i].index_fill_(0, idx_change, False)
    #print(neg_mask_k1)
    # print(torch.sum(neg_mask_k1))

    # print('----')
    neg_mask_k2 = (iou>=0.167) & (iou<0.332)
    #最后处理小iou的 可以补充数量
    for i in range(len(neg_mask_k2)):
        idx = torch.where(neg_mask_k2[i]==True)[0]
        #print(len(idx), sample_every_bin[0], sample_every_bin[0].item())
        change_num = len(idx)-int(sample_every_bin[i].item())
        if change_num>0:
            idx_change = idx[torch.randperm(len(idx))[:change_num]]
            neg_mask_k2[i].index_fill_(0, idx_change, False)
    # print(torch.sum(neg_mask_k2))

    exist_num = torch.sum(neg_mask_k1,dim=-1)+torch.sum(neg_mask_k2,dim=-1)
    # print(exist_num)
    # print('----===')
    neg_mask_k3 = (iou>=0) & (iou<0.167)
    #最后处理小iou的 可以补充数量
    for i in range(len(neg_mask_k3)):
        idx = torch.where(neg_mask_k3[i]==True)[0]
        #print(len(idx), sample_every_bin[0], sample_every_bin[0].item())
        change_num = len(idx)-(int(sample_every_bin[i].item())*3-int(exist_num[i].item()))
        if change_num>0:
            idx_change = idx[torch.randperm(len(idx))[:change_num]]
            neg_mask_k3[i].index_fill_(0, idx_change, False)
    # print(torch.sum(neg_mask_k3))
    #print(torch.sum(neg_mask_k3,dim=-1))

    neg_mask = neg_mask_k1 | neg_mask_k2 | neg_mask_k3
    # print(neg_mask.shape)
    # print(torch.sum(neg_mask))
    # b

    
    # iou_np = iou.cpu().numpy()
    # print(iou_np.shape)
    # print(np.min(iou_np))#-1.0
    # print(np.max(iou_np))#0.47
    # print(np.sum(iou_np==-1))#691 == num_pos
    # print(num_pos,sum(num_pos))
    # b



    #shape都是32*4420，求或，有一个true就保留
    return pos_mask | neg_mask
