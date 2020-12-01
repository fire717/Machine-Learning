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

net_work_path = './yufacedetectnet-open-v1.prototxt'
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

def sparseOutput(output, conf_thresh=0.5, topn=10):
        # self.net.blobs['data'].data[...] = self.preprocess(img).transpose((2, 0, 1)) 
        # output = self.net.forward()['detection_out']
        det_label = output[0,0,:,1]
        det_conf = output[0,0,:,2]
        det_xmin = output[0,0,:,3]
        det_ymin = output[0,0,:,4]
        det_xmax = output[0,0,:,5]
        det_ymax = output[0,0,:,6]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            result.append([xmin, ymin, xmax, ymax, label, score])
        return result


for image in image_list:
    print("Processing file: {}".format(image))
    img = cv2.imread(images_dir + image)
   # print(img[0][:10])

    origin_h, origin_w,_ = img.shape
    print(img.shape)

    w = 320
    h = 240
   #  resize_mat = cv2.resize(img, (w, h),)

   #  #cv2.imwrite(result_dir + "gray.png", res)

   #  #res = res.reshape((1,3,w,h))
   # # res2 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB);

    
   #  #resize_mat = resize_mat.T
   #  # print(resize_mat[:,0,0],resize_mat[:,0,1])
   #  new_img = resize_mat.copy()
   #  new_img = np.float32(new_img)
   #  new_img [0,:,:]= new_img[0,:,:] - 104
   #  new_img [1,:,:]= new_img[1,:,:] - 117
   #  new_img [2,:,:]= new_img[2,:,:] - 123
    new_img = preprocess(img,h,w).transpose((2, 0, 1))
    print(new_img.shape)
    #new_img = new_img
    # print(new_img.shape)
    # print(new_img[:,0,0],new_img[:,0,1])
    # b

   # net.blobs['data'].reshape(1, 3, w, h)
    net.blobs['data'].data[...] = new_img

    t = time.time()
    out = net.forward()
    print("time: ", time.time() - t)
    #print("out: ", out)
    # print("out: ", out['conv3_3_norm_mbox_loc_perm'].shape)#('out: ', (1, 30, 40, 12))
    # print("out: ", out['conv3_3_norm_mbox_conf_perm'].shape)#('out: ', (1, 30, 40, 6))
   
    print("out: ", out['detection_out'].shape)#(1, 1, 23, 7)
    print("out['detection_out'][0][0] ", out['detection_out'][0][0].shape)
    #print("out: ", out['conv3_3_norm_mbox_conf_perm'].shape)


#[image_id, label, confidence, xmin, ymin, xmax, ymax]


  
    res = sparseOutput(out['detection_out'])
    print("res ", res)

    # for line in out['detection_out'][0][0]:
    #     print(line.tolist())
    #     assert len(line)==7
    #     x0 = int(line[3]*w)
    #     y0 = int(line[4]*h)
    #     x1 = int(line[5]*w)
    #     y1 = int(line[6]*h)
    #     cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),2)
    for line in res:
        print(line)

        x0 = int(line[0]*origin_w)
        y0 = int(line[1]*origin_h)
        x1 = int(line[2]*origin_w)
        y1 = int(line[3]*origin_h)
        cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),2)

    cv2.imwrite("res.jpg", img)
