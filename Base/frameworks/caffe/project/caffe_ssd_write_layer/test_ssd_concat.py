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

net_work_path = './yufacedetectnet-open-v1-concat.prototxt'
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


def concat_layer(input_data1, input_data2, input_data3, input_data4):

    if input_data1 is None or input_data2 is None or input_data3 is None or input_data4 is None:
        print(": The input data is null.")
        return -1

    output_data = np.concatenate((input_data1, input_data2, input_data3, input_data4), axis = -1)
    return output_data

    # facesInfo = 1
    # return facesInfo

for image in image_list:
    print("Processing file: {}".format(image))
    img = cv2.imread(images_dir + image)
   # print(img[0][:10])

    origin_h, origin_w,_ = img.shape
    print(img.shape)

    w = 320
    h = 240

    new_img = preprocess(img,h,w).transpose((2, 0, 1))
    print(new_img.shape)

    net.blobs['data'].data[...] = new_img

    t = time.time()
    out = net.forward()
    print("time: ", time.time() - t)

    print("conv3_3_norm_mbox_priorbox: ", out['conv3_3_norm_mbox_priorbox'][0,0,:10])
    print("conv3_3_norm_mbox_priorbox: ", out['conv3_3_norm_mbox_priorbox'][0,1,:10])
    print("conv3_3_norm_mbox_loc_flat: ", out['conv3_3_norm_mbox_loc_flat'][0,:10])
    print("conv3_3_norm_mbox_conf_flat: ", out['conv3_3_norm_mbox_conf_flat'][0,:10])

    print("conv6_3_norm_mbox_priorbox: ", out['conv6_3_norm_mbox_priorbox'][0,0,:10])
    print("conv6_3_norm_mbox_priorbox: ", out['conv6_3_norm_mbox_priorbox'][0,1,:10])
    print("conv6_3_norm_mbox_loc_flat: ", out['conv6_3_norm_mbox_loc_flat'][0,:10])
    print("conv6_3_norm_mbox_conf_flat: ", out['conv6_3_norm_mbox_conf_flat'][0,:10])

    # print("conv3_3_norm_mbox_priorbox: ", out['conv3_3_norm_mbox_priorbox'].shape)
    # print("conv3_3_norm_mbox_loc_flat: ", out['conv3_3_norm_mbox_loc_flat'].shape)
    # print("conv3_3_norm_mbox_conf_flat: ", out['conv3_3_norm_mbox_conf_flat'].shape)
    
    # print("conv4_3_norm_mbox_priorbox: ", out['conv4_3_norm_mbox_priorbox'].shape)
    # print("conv4_3_norm_mbox_loc_flat: ", out['conv4_3_norm_mbox_loc_flat'].shape)
    # print("conv4_3_norm_mbox_conf_flat: ", out['conv4_3_norm_mbox_conf_flat'].shape)

    # print("conv5_3_norm_mbox_priorbox: ", out['conv5_3_norm_mbox_priorbox'].shape)
    # print("conv5_3_norm_mbox_loc_flat: ", out['conv5_3_norm_mbox_loc_flat'].shape)
    # print("conv5_3_norm_mbox_conf_flat: ", out['conv5_3_norm_mbox_conf_flat'].shape)

    # print("conv6_3_norm_mbox_priorbox: ", out['conv6_3_norm_mbox_priorbox'].shape)
    # print("conv6_3_norm_mbox_loc_flat: ", out['conv6_3_norm_mbox_loc_flat'].shape)
    # print("conv6_3_norm_mbox_conf_flat: ", out['conv6_3_norm_mbox_conf_flat'].shape)
#     ('conv3_3_norm_mbox_priorbox: ', (1, 2, 14400))
# ('conv3_3_norm_mbox_loc_flat: ', (1, 14400))
# ('conv3_3_norm_mbox_conf_flat: ', (1, 7200))
# ('conv4_3_norm_mbox_priorbox: ', (1, 2, 2400))
# ('conv4_3_norm_mbox_loc_flat: ', (1, 2400))
# ('conv4_3_norm_mbox_conf_flat: ', (1, 1200))
# ('conv5_3_norm_mbox_priorbox: ', (1, 2, 640))
# ('conv5_3_norm_mbox_loc_flat: ', (1, 640))
# ('conv5_3_norm_mbox_conf_flat: ', (1, 320))
# ('conv6_3_norm_mbox_priorbox: ', (1, 2, 240))
# ('conv6_3_norm_mbox_loc_flat: ', (1, 240))
# ('conv6_3_norm_mbox_conf_flat: ', (1, 120))

    print("out['conv3_3_norm_mbox_priorbox']: ", out['conv3_3_norm_mbox_priorbox'][0,0,:10])
    #[-0.003125  , -0.00416667,  0.028125  ,  0.0375    , -0.0125    ,-0.01666667,  0.0375    ,  0.05      , -0.025     , -0.03333334]
    mbox_priorbox = concat_layer(out['conv3_3_norm_mbox_priorbox'], out['conv4_3_norm_mbox_priorbox'],
                                                    out['conv5_3_norm_mbox_priorbox'],  out['conv6_3_norm_mbox_priorbox'])
    mbox_loc = concat_layer(out['conv3_3_norm_mbox_loc_flat'], out['conv4_3_norm_mbox_loc_flat'],
                                                    out['conv5_3_norm_mbox_loc_flat'],  out['conv6_3_norm_mbox_loc_flat'])
    mbox_conf_flatten = concat_layer(out['conv3_3_norm_mbox_conf_flat'], out['conv4_3_norm_mbox_conf_flat'],
                                                    out['conv5_3_norm_mbox_conf_flat'],  out['conv6_3_norm_mbox_conf_flat'])
    print("mbox_priorbox: ", mbox_priorbox.shape)  #(1, 2, 17680))
    print("mbox_loc: ", mbox_loc.shape) #(1, 17680))
    print("mbox_conf_flatten: ", mbox_conf_flatten.shape) #(1, 8840))


    res = detection_output(mbox_priorbox,  mbox_loc, mbox_conf_flatten)
    for line in res:
        print(line)

        x0 = int(line[1][0]*origin_w)
        y0 = int(line[1][1]*origin_h)
        x1 = int(line[1][2]*origin_w)
        y1 = int(line[1][3]*origin_h)
        cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),2)

    cv2.imwrite("test_img2.jpg",img)