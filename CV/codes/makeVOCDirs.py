import os



BASE_PATH = "./"

voc_dir = os.path.join(BASE_PATH, "VOC2007")

anno_dir = os.path.join(voc_dir, "Annotations")
set_dir = os.path.join(voc_dir, "ImageSets")
img_dir = os.path.join(voc_dir, "JPEGImages")

main_dir = os.path.join(set_dir, "Main")


if not os.path.exists(voc_dir):
    os.makedirs(voc_dir)

if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)

if not os.path.exists(set_dir):
    os.makedirs(set_dir)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(main_dir):
    os.makedirs(main_dir)
