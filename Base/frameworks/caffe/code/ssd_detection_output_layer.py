def JaccardOverlap(bbox1, bbox2):

    def _IntersectBBox(bbox1, bbox2) :
        intersect_bbox = [0,0,0,0] #[xmin, ymin, xmax, ymax]
        if (bbox2[0] > bbox1[2] or bbox2[2]< bbox1[0] or bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1]) :
            #Return [0, 0, 0, 0] if there is no intersection.
           return intersect_bbox
        else:
            intersect_bbox[0] = max(bbox1[0], bbox2[0])
            intersect_bbox[1] = max(bbox1[1], bbox2[1])
            intersect_bbox[2] = min(bbox1[2], bbox2[2])
            intersect_bbox[3] = min(bbox1[3], bbox2[3])
            return intersect_bbox

    intersect_bbox = _IntersectBBox(bbox1, bbox2)
    intersect_width = intersect_bbox[2] - intersect_bbox[0]
    intersect_height = intersect_bbox[3]- intersect_bbox[1]

    if (intersect_width > 0 and intersect_height > 0) :
        intersect_size = intersect_width * intersect_height
        bsize1 = (bbox1[2]- bbox1[0])*(bbox1[3] - bbox1[1])
        bsize2 = (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1])
        return intersect_size*1.0 / ( bsize1 + bsize2 - intersect_size)
    else :
        return 0

def detection_output(mbox_priorbox, mbox_loc, mbox_conf, overlap_threshold=0.3, 
                                confidence_threshold=0.5, top_k=100, keep_top_k=50,):

    assert mbox_priorbox.shape[-1]==mbox_loc.shape[-1]
    assert mbox_priorbox.shape[-1]==mbox_conf.shape[-1]*2

    prior_variance = [0.1, 0.1, 0.2, 0.2];
    pPriorBox = mbox_priorbox[0];
    pLoc = mbox_loc[0];
    pConf = mbox_conf[0];

    score_bbox_vec = []

    i = 1
    while i < mbox_conf.shape[-1]:
        if  pConf[i]>confidence_threshold:
            fx1 = pPriorBox[0][i * 2 - 2]
            fy1 = pPriorBox[0][i * 2 - 1]
            fx2 = pPriorBox[0][i * 2]
            fy2 = pPriorBox[0][i * 2 + 1]

            locx1 = pLoc[i * 2 - 2]
            locy1 = pLoc[i * 2 - 1]
            locx2 = pLoc[i * 2]
            locy2 =pLoc[i * 2 + 1]

            prior_width = fx2 - fx1
            prior_height = fy2 - fy1
            prior_center_x = (fx1 + fx2)/2.0
            prior_center_y = (fy1 + fy2)/2.0

            box_centerx = prior_variance[0] * locx1 * prior_width + prior_center_x
            box_centery = prior_variance[1] * locy1 * prior_height + prior_center_y
            box_width =  (np.e**(prior_variance[2] * locx2)) * prior_width
            box_height =(np.e**(prior_variance[3] * locy2)) * prior_height

            fx1 = box_centerx - box_width / 2.0
            fy1 = box_centery - box_height /2.0
            fx2 = box_centerx + box_width / 2.0
            fy2 = box_centery + box_height /2.0

            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(1., fx2)
            fy2 = min(1., fy2)

            xmin = fx1;
            ymin = fy1;
            xmax = fx2;
            ymax = fy2;
            bbox = [xmin, ymin, xmax, ymax]
    #         score_bbox_vec.push_back(std::make_pair(pConf[i], bb));

            score_bbox_vec.append([pConf[i], bbox])
        i+=2

    
    score_bbox_vec = sorted(score_bbox_vec, key = lambda x: x[0], reverse = True)
    #Keep top_k scores if needed.
    if (top_k > -1 and top_k < len(score_bbox_vec)):
        score_bbox_vec = score_bbox_vec[:top_k]

    final_score_bbox_vec = []
    #print(len(final_score_bbox_vec), len(score_bbox_vec))
    #NMS
    for i in range(len(score_bbox_vec)):
        bb1 = score_bbox_vec[i][1]
        keep_flag = True
        for j in range(len(final_score_bbox_vec)):
            if keep_flag:
                bb2 = final_score_bbox_vec[j][1]
                overlap = JaccardOverlap(bb1, bb2)
                keep_flag = (overlap<= overlap_threshold)
            else:
                break
        if keep_flag:
            final_score_bbox_vec.append(score_bbox_vec[i])

    if keep_top_k > -1 and keep_top_k < len(final_score_bbox_vec):
        final_score_bbox_vec = final_score_bbox_vec[:keep_top_k]

    #print(len(final_score_bbox_vec))
    return final_score_bbox_vec
