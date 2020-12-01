


import os
import random

def getAllName(file_dir, tail_list = ['.jpg']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L




# make all
img_names = getAllName("VOC2007/JPEGImages")
with open("VOC2007/ImageSets/Main/trainval.txt", "w", encoding="utf-8") as f:
    for img_name in img_names:
        f.write(os.path.basename(img_name)[:-4]+"\n")


batch_size = 16
# split
split_ratio = 0.1
with open("VOC2007/ImageSets/Main/trainval.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
print("total label: ", len(lines))
print("batch  size: ", batch_size)
print("train steps: ", int(len(lines)*(1-split_ratio)*1.0/batch_size))

f_train = open("VOC2007/ImageSets/Main/train.txt", "w", encoding="utf-8")
f_val  = open("VOC2007/ImageSets/Main/val.txt", "w", encoding="utf-8")

for line in lines:
    if random.random() < split_ratio:
        f_val.write(line)

    else:
        f_train.write(line)

f_train.close()
f_val.close()
