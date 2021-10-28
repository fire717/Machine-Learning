
import glob
import cv2

xml_head = '''<annotation>
    <folder>VOC2007</folder>
    <filename>{}</filename>.
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
    </source>   
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''
xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''

#--data
#----train 训练集图片
#----train_txt 对应的txt标签
#----train_xml 对应的xml标签

root='./'


labels = {0: 'person'}

txt_Lists = glob.glob(root +'labels_abs'+ '/*.txt')
print(len(txt_Lists))
# print(txt_Lists)
cnt=0

for txt_path in txt_Lists:
    filename=txt_path.split('\\')
    filename=filename[-1]
    filename=filename.split('.')
    filename=filename[0]

    txt = root+'labels_abs/'+filename+'.txt'
    # jpg=root+'train/'+filename+'.jpg' #jpg path
    xml=root+'labels_xml/'+filename+'.xml'

    print(txt)
    print(xml)

    obj = ''

    # img = cv2.imread(jpg)
    img_h, img_w = 1080, 1920

    print('h_factor:',img_h,'  w_factor:',img_w)
    # cv2.imshow("img", img)  #显示图片
    # cv2.waitKey(0)
    # cv2.destroyWindow("img")

    head = xml_head.format(str(filename), str(img_w), str(img_h), "3")

    with open(txt, 'r') as f:
        for line in f.readlines():
            yolo_datas = line.strip().split(' ')
            label = int(float(yolo_datas[0].strip()))
            # center_x = round(float(str(yolo_datas[1]).strip()) * img_w)
            # center_y = round(float(str(yolo_datas[2]).strip()) * img_h)
            # bbox_width = round(float(str(yolo_datas[3]).strip()) * img_w)
            # bbox_height = round(float(str(yolo_datas[4]).strip()) * img_h)

            # xmin = str(int(center_x - bbox_width / 2))
            # ymin = str(int(center_y - bbox_height / 2))
            # xmax = str(int(center_x + bbox_width / 2))
            # ymax = str(int(center_y + bbox_height / 2))

            xmin = str(int(float(yolo_datas[2].strip())))
            ymin = str(int(float(yolo_datas[3].strip())))
            xmax = str(int(float(yolo_datas[4].strip())))
            ymax = str(int(float(yolo_datas[5].strip())))

            obj += xml_obj.format(labels[label], xmin, ymin, xmax, ymax)

    with open(xml, 'w') as f_xml:
        f_xml.write(head + obj + xml_end)
    cnt += 1
    print(cnt)
