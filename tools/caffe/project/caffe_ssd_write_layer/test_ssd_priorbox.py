import sys
import os

import time
import pprint

import caffe

import cv2
import numpy as np

def file_list_fn(path):

    file_list = []
    files = os.listdir(path)
    for f in files:
        file_list.append(f)
    return file_list

net_work_path = './yufacedetectnet-open-v1-priorbox.prototxt'
weight_path = './yufacedetectnet-open-v1.caffemodel'
images_dir = './img/'
result_dir = './out/'

image_list = file_list_fn(images_dir)
caffe.set_mode_cpu()
net = caffe.Net(net_work_path, weight_path, caffe.TEST)
#net.name = 'FaceThink_face_landmark_test'


total_landmark_time = 0.0
face_total = 0.0
#points =

def preprocess(src,h,w):
        img = cv2.resize(src, (w,h))
        img = np.array(img, dtype=np.float32)
        img -= np.array((104, 117, 123)) 
        return img

# def sparseOutput(output, conf_thresh=0.5, topn=10):
#         # self.net.blobs['data'].data[...] = self.preprocess(img).transpose((2, 0, 1)) 
#         # output = self.net.forward()['detection_out']
#         det_label = output[0,0,:,1]
#         det_conf = output[0,0,:,2]
#         det_xmin = output[0,0,:,3]
#         det_ymin = output[0,0,:,4]
#         det_xmax = output[0,0,:,5]
#         det_ymax = output[0,0,:,6]
#         top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
#         top_conf = det_conf[top_indices]
#         top_label_indices = det_label[top_indices].tolist()
#         top_xmin = det_xmin[top_indices]
#         top_ymin = det_ymin[top_indices]
#         top_xmax = det_xmax[top_indices]
#         top_ymax = det_ymax[top_indices]

#         result = []
#         for i in range(min(topn, top_conf.shape[0])):
#             xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
#             ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
#             xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
#             ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
#             score = top_conf[i]
#             label = int(top_label_indices[i])
#             result.append([xmin, ymin, xmax, ymax, label, score])
#         return result

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
    c = 0 #print top 10
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
            # if c<10:
            #     print(" --------- -----bbox : ", [int(x*300) for x in bbox])
            #     c+=1
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


def concat_layer(input_data1, input_data2, input_data3, input_data4):

    if input_data1 is None or input_data2 is None or input_data3 is None or input_data4 is None:
        print(": The input data is null.")
        return -1

    output_data = np.concatenate((input_data1, input_data2, input_data3, input_data4), axis = -1)
    return output_data

def prior_box_layer(feature_width,feature_height,imageData,  pWinSizes):
    num_sizes = len(pWinSizes)
    # feature_width = featureData.shape[3]
    # feature_height = featureData.shape[2]
    image_width = imageData.shape[2]
    image_height = imageData.shape[1]

    step_w = image_width*1.0 / feature_width
    step_h = image_height*1.0/ feature_height
    print("---step: ", step_w," ",step_h)
    outputData = []
    for h in range(feature_height):
        h_tmp = []
        for w in range(feature_width) :
            pOut = []
            for s in range(num_sizes): 
                min_size_ = pWinSizes[s]
                box_width = box_height = min_size_
                
                center_x = w * step_w + step_w / 2.0
                center_y = h * step_h + step_h / 2.0

               # print(min_size_, box_width, center_x, center_y, image_width)
                #b
                # xmin
                pOut.append( (center_x - box_width / 2.0) / image_width)
                # ymin
                pOut.append( (center_y - box_height / 2.0) / image_height)
                # xmax
                pOut.append( (center_x + box_width / 2.0) / image_width)
                # ymax
                pOut.append( (center_y + box_height / 2.0) / image_height)
            # if h== 0 and w==0:
            #     print("prior box h:   ",h)
            #     print("prior box w:   ",w)
            #     print("prior box min_size_:   ",min_size_)
            #     print("-------prior box image_width:   ",image_width)
            #     print("-------prior box image_height:   ",image_height)
            #     print("-------prior box box_width:   ",box_width)
            #     print("-------prior box box_height:   ",box_height)

            h_tmp.append(pOut)
        outputData.append(h_tmp)

    return np.array(outputData)

    # facesInfo = 1
    # return facesInfo
def softmax1vector2class(inputOutputData):
    num = inputOutputData.shape[-1]
    pData = inputOutputData[0]

    for i in range(0,num,2):
        v1 = pData[i]
        v2 = pData[i+1]
        vm = max(v1, v2)
        v1 -= vm
        v2 -= vm
        v1 = np.e**(v1)
        v2 = np.e**(v2)
        vm = v1 + v2;
        pData[i] = v1/vm;
        pData[i+1] = v2/vm;
    return np.array([pData])

for image in image_list:
    print("Processing file: {}".format(image))
    img = cv2.imread(images_dir + image)
   # print(img[0][:10])

    origin_h, origin_w,_ = img.shape
    print(img.shape)

    w = 300
    h = 300

    new_img = preprocess(img,h,w).transpose((2, 0, 1))
    print(new_img.shape)

    net.blobs['data'].data[...] = new_img

    t = time.time()
    out = net.forward()
    print("time: ", time.time() - t)

    print("conv3_3_norm_mbox_loc_perm: ", out['conv3_3_norm_mbox_loc_perm'].shape)
    print("conv3_3_norm_mbox_conf_perm: ", out['conv3_3_norm_mbox_conf_perm'].shape)
    conv3_3_norm_mbox_loc_flat = np.reshape(out['conv3_3_norm_mbox_loc_perm'], (1,-1))
    conv3_3_norm_mbox_conf_flat = np.reshape(out['conv3_3_norm_mbox_conf_perm'], (1,-1))
    # print("conv3_3_norm_mbox_loc_flat: ",conv3_3_norm_mbox_loc_flat.shape)
    # print("conv3_3_norm_mbox_conf_flat: ", conv3_3_norm_mbox_conf_flat.shape)
    print("conv3_3_norm_mbox_loc_flat 10----------:", [x for x in conv3_3_norm_mbox_loc_flat[0][:8]])
    #print("conv3_3_norm_mbox_loc_flat ----------:", [int(x*300) for x in conv3_3_norm_mbox_loc_flat[0][:8]])
    print("conv3_3_norm_mbox_conf_flat 10----------:", [x for x in conv3_3_norm_mbox_conf_flat[0][:8]])
    #print("conv3_3_norm_mbox_conf_flat ----------:", [int(x*300) for x in conv3_3_norm_mbox_conf_flat[0][:8]])

    print("conv4_3_norm_mbox_loc_perm: ", out['conv4_3_norm_mbox_loc_perm'].shape)
    print("conv4_3_norm_mbox_conf_perm: ", out['conv4_3_norm_mbox_conf_perm'].shape)
    conv4_3_norm_mbox_loc_flat = np.reshape(out['conv4_3_norm_mbox_loc_perm'], (1,-1))
    conv4_3_norm_mbox_conf_flat = np.reshape(out['conv4_3_norm_mbox_conf_perm'], (1,-1))
    print("conv4_3_norm_mbox_loc_flat 10----------:", [x for x in conv4_3_norm_mbox_loc_flat[0][:8]])
    #print("conv4_3_norm_mbox_loc_flat ----------:", [int(x*300) for x in conv4_3_norm_mbox_loc_flat[0][:8]])
    print("conv4_3_norm_mbox_conf_flat 10----------:", [x for x in conv4_3_norm_mbox_conf_flat[0][:8]])
    #print("conv4_3_norm_mbox_conf_flat ----------:", [int(x) for x in conv4_3_norm_mbox_conf_flat[0][:8]])

    print("conv5_3_norm_mbox_loc_perm: ", out['conv5_3_norm_mbox_loc_perm'].shape)
    print("conv5_3_norm_mbox_conf_perm: ", out['conv5_3_norm_mbox_conf_perm'].shape)
    conv5_3_norm_mbox_loc_flat = np.reshape(out['conv5_3_norm_mbox_loc_perm'], (1,-1))
    conv5_3_norm_mbox_conf_flat = np.reshape(out['conv5_3_norm_mbox_conf_perm'], (1,-1))
    print("conv5_3_norm_mbox_loc_flat 10----------:", [x for x in conv5_3_norm_mbox_loc_flat[0][:8]])
    #print("conv5_3_norm_mbox_loc_flat ----------:", [int(x*300) for x in conv5_3_norm_mbox_loc_flat[0][:8]])
    print("conv5_3_norm_mbox_conf_flat 10----------:", [x for x in conv5_3_norm_mbox_conf_flat[0][:8]])
    #print("conv5_3_norm_mbox_conf_flat ----------:", [int(x*300) for x in conv5_3_norm_mbox_conf_flat[0][:8]])

    print("conv6_3_norm_mbox_loc_perm: ", out['conv6_3_norm_mbox_loc_perm'].shape)
    print("conv6_3_norm_mbox_conf_perm: ", out['conv6_3_norm_mbox_conf_perm'].shape)
    conv6_3_norm_mbox_loc_flat = np.reshape(out['conv6_3_norm_mbox_loc_perm'], (1,-1))
    conv6_3_norm_mbox_conf_flat = np.reshape(out['conv6_3_norm_mbox_conf_perm'], (1,-1))
    print("conv6_3_norm_mbox_loc_flat 10----------:", [x for x in conv6_3_norm_mbox_loc_flat[0][:8]])
    #print("conv6_3_norm_mbox_loc_flat ----------:", [int(x*300) for x in conv6_3_norm_mbox_loc_flat[0][:8]])
    print("conv6_3_norm_mbox_conf_flat 10----------:", [x for x in conv6_3_norm_mbox_conf_flat[0][:8]])
    #print("conv6_3_norm_mbox_conf_flat ----------:", [int(x*300) for x in conv6_3_norm_mbox_conf_flat[0][:8]])

    # Prior Box
    
    out = net.forward(end='conv3_3_norm')
   # print("conv3_3_norm: ", out['conv3_3_norm'].shape)#(1, 32, 30, 40)) 240       //300  (1, 32, 38, 38)
    pWinSizes = [10,16,24]
    conv3priorbox = prior_box_layer( 38,38, new_img,  pWinSizes)
    #print("conv3priorbox: ", conv3priorbox.shape)# (38, 38, 12))    12=3size * 4xxyy
    conv3priorbox = np.reshape(conv3priorbox, (1,-1))
    #print("conv3priorbox: ", conv3priorbox.shape)# (1, 17328)
    #print("conv3priorbox vvv: ", conv3priorbox[0][:10])

    out = net.forward(end='conv4_3_norm')
    #print("conv4_3_norm: ", out['conv4_3_norm'].shape)#(1, 64, 15, 20)) 240       //300  (1, 64, 19, 19)
    pWinSizes = [32,48]
    conv4priorbox = prior_box_layer( 19,19, new_img,  pWinSizes)
   # print("conv4priorbox: ", conv4priorbox.shape)#(19, 19, 8)
    conv4priorbox = np.reshape(conv4priorbox, (1,-1))
   # print("conv4priorbox: ", conv4priorbox.shape)# (1, 2888)

    out = net.forward(end='conv5_3_norm')
    #print("conv5_3_norm: ", out['conv5_3_norm'].shape)#(1, 128, 8, 10)) 240       //300  (1, 128, 10, 10)
    pWinSizes = [64,96]
    conv5priorbox = prior_box_layer( 10,10, new_img,  pWinSizes)
    #print("conv5priorbox: ", conv5priorbox.shape)  #(10, 10, 8)
    conv5priorbox = np.reshape(conv5priorbox, (1,-1))
    #print("conv5priorbox: ", conv5priorbox.shape)# (1, 800)

    out = net.forward(end='conv6_3_norm')
   # print("conv6_3_norm: ", out['conv6_3_norm'].shape)#(1,128, 4, 5)) 240       //300   (1, 128, 5, 5))
    pWinSizes = [128,192,256]
    conv6priorbox = prior_box_layer( 5,5, new_img,  pWinSizes)
    #print("conv6priorbox: ", conv6priorbox.shape) #(5, 5, 12)
    conv6priorbox = np.reshape(conv6priorbox, (1,-1))
   # print("conv6priorbox: ", conv6priorbox.shape)# (1, 300)


    #print("conv3priorbox t: ", conv3priorbox[0,:10])
    #[-0.003125  , -0.00416667,  0.028125  ,  0.0375    , -0.0125    ,-0.01666667,  0.0375    ,  0.05      , -0.025     , -0.03333334]
    #300 [-0.00350877, -0.00350877,  0.02982456,  0.02982456, -0.01350877, -0.01350877,  0.03982456,  0.03982456, -0.02684211, -0.02684211]

### concat
    mbox_priorbox = concat_layer(conv3priorbox, conv4priorbox,
                                                    conv5priorbox,  conv6priorbox)
    mbox_loc = concat_layer(conv3_3_norm_mbox_loc_flat, conv4_3_norm_mbox_loc_flat,
                                                    conv5_3_norm_mbox_loc_flat,  conv6_3_norm_mbox_loc_flat)
    mbox_conf_flatten = concat_layer(conv3_3_norm_mbox_conf_flat, conv4_3_norm_mbox_conf_flat,
                                                    conv5_3_norm_mbox_conf_flat, conv6_3_norm_mbox_conf_flat)
    #print("mbox_priorbox: ", mbox_priorbox.shape)  #(1, 2, 21316))
   # print("mbox_loc: ", mbox_loc.shape) #(1, 21316))
    #print("mbox_conf_flatten: ", mbox_conf_flatten.shape) #(1, 10658))   300*300



   # print("before softmax1vector2class: ", mbox_conf_flatten[0][:10])
    mbox_conf_flatten = softmax1vector2class(mbox_conf_flatten)
    #print("after softmax1vector2class: ", mbox_conf_flatten[0][:10])

    mbox_priorbox = np.concatenate((mbox_priorbox,mbox_priorbox),-1)
    mbox_priorbox = np.reshape(mbox_priorbox, (1,2,-1))
    #print("mbox_priorbox: ", mbox_priorbox.shape)  #(1, 2, 17680))   320*240
    #print("mbox_loc: ", mbox_loc.shape) #(1, 17680))
    #print("mbox_conf_flatten: ", mbox_conf_flatten.shape) #(1, 8840))

#####  detection_output
    res = detection_output(mbox_priorbox,  mbox_loc, mbox_conf_flatten)
    for line in res:
        print(line)

        x0 = int(line[1][0]*origin_w)
        y0 = int(line[1][1]*origin_h)
        x1 = int(line[1][2]*origin_w)
        y1 = int(line[1][3]*origin_h)
        cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),2)

    cv2.imwrite("test_img3.jpg",img)