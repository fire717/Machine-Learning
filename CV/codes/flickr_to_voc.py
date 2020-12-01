import os
import shutil
import cv2




def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件



BASE_PATH = "./flickr_logos_27_dataset"
VOC_PATH = "./VOC2007"

label_path = os.path.join(BASE_PATH, "flickr_logos_27_dataset_training_set_annotation.txt")
img_path = os.path.join(BASE_PATH, "flickr_logos_27_dataset_images")

needed_labels = ['Adidas', 'Nike', 'Puma'] 
trainval_list = []
xml_dict = {}

with open(label_path, "r") as f:
    lines = f.readlines()
    print("total: ", len(lines))
    lines = [line.replace(" 2 "," 1 ") for line in lines]
    lines = [line.replace(" 3 "," 1 ") for line in lines]
    lines = [line.replace(" 4 "," 1 ") for line in lines]
    lines = [line.replace(" 5 "," 1 ") for line in lines]
    lines = [line.replace(" 6 "," 1 ") for line in lines]
    lines = list(set(lines))
    print("total set: ", len(lines))

    for line in lines:
        if " 1 " not in line:
            continue
        img_name, label_name, label_id, x1, y1, x2, y2 = line.strip().split(" ")
        base_name = img_name.split('.')[0]

        if label_name in needed_labels:

            #1.add label to txt
            trainval_list.append(base_name)

            #2.copy img
            img_read_path = os.path.join(img_path, img_name)
            img_save_path = os.path.join(VOC_PATH, "JPEGImages", img_name)
            mycopyfile(img_read_path, img_save_path)

            #3.add xml dict
            if base_name in xml_dict:
                xml_dict[base_name] = xml_dict[base_name] + [[label_name,x1,y1,x2,y2]]
            else:
                xml_dict[base_name] = [[label_name,x1,y1,x2,y2]]

trainval_list = list(set(trainval_list))
print("done: ", len(trainval_list), len(xml_dict))

#generate txt
with open(os.path.join(VOC_PATH,"ImageSets","Main","trainval.txt"), "w", encoding="utf-8") as f:
    for line in trainval_list:
        f.write(line+"\n")

#generate xml
import xml.dom.minidom as minidom


for k,v in xml_dict.items():
    dom = minidom.getDOMImplementation().createDocument(None,'annotation',None)
    root = dom.documentElement
    #print(k,v)
    
    img = cv2.imread(os.path.join(img_path, k+".jpg"))
    h,w,c = img.shape

    element = dom.createElement('filename')
    element.appendChild(dom.createTextNode(k+".jpg"))
    root.appendChild(element)

    element = dom.createElement('size')
    element_son = dom.createElement('width')
    element_son.appendChild(dom.createTextNode(str(w)))
    element.appendChild(element_son)
    element_son = dom.createElement('height')
    element_son.appendChild(dom.createTextNode(str(h)))
    element.appendChild(element_son)
    element_son = dom.createElement('depth')
    element_son.appendChild(dom.createTextNode(str(c)))
    element.appendChild(element_son)
    root.appendChild(element)

    for i,box in enumerate(v):
        label_name, x1, y1, x2, y2 = box

        element = dom.createElement('object')
        element_son = dom.createElement('name')
        element_son.appendChild(dom.createTextNode(label_name))
        element.appendChild(element_son)

        element_son = dom.createElement('truncated')
        element_son.appendChild(dom.createTextNode("0"))
        element.appendChild(element_son)

        element_son = dom.createElement('difficult')
        element_son.appendChild(dom.createTextNode("0"))
        element.appendChild(element_son)

        element_son = dom.createElement('bndbox')
        element_grandson = dom.createElement('xmin')
        element_grandson.appendChild(dom.createTextNode(x1))
        element_son.appendChild(element_grandson)
        element_grandson = dom.createElement('ymin')
        element_grandson.appendChild(dom.createTextNode(y1))
        element_son.appendChild(element_grandson)
        element_grandson = dom.createElement('xmax')
        element_grandson.appendChild(dom.createTextNode(x2))
        element_son.appendChild(element_grandson)
        element_grandson = dom.createElement('ymax')
        element_grandson.appendChild(dom.createTextNode(y2))
        element_son.appendChild(element_grandson)

        element.appendChild(element_son)
        root.appendChild(element)


    # 保存文件
    with open(os.path.join(VOC_PATH,"Annotations",k+'.xml'), 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n',encoding='utf-8')
    
