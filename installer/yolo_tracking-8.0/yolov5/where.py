import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from  PIL import Image, ImageDraw ,ImageFont

def cv2ImgAddText(image, text, left, top, textColor=(0,255,0), textSize=20):
    if(isinstance(image,np.ndarray)):
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    draw = ImageDraw.Draw(image)
    fontStyle = ImageFont.truetype("/usr/share/fonts/truetype/arphic/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font = fontStyle)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def draw_gt_multi(detect_path, gt_folder_path):
    images = os.listdir(detect_path)

    # labels = ["1", "2", "3", "4", "5", "7", "8", "9", "10", "12", "15", "16", "22", "23", "24", ]
    # names = ["枪支","电击器","警棍","手铐","管制刀具","电子烟花","礼花","烟花","烟饼","压缩罐","空包弹","霰弹", "块状电池", "节状电池", "纽扣电池",]

    for image in images:
        if not os.path.isdir(image):
            basename = os.path.basename(image)
            gt_path = os.path.join(gt_folder_path, basename[:-4] + '.xml')
            image = cv2.imread(os.path.join(detect_path, image))

            tree = ET.parse(gt_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text.split('_')[0].split('-')[0]
                # print(class_name)
                if class_name == "2":
                    print(gt_path)
                # class_name = names[labels.index(class_name)]
                # bbox = obj.find('bndbox')
                # xmin = int(bbox.find('xmin').text)
                # ymin = int(bbox.find('ymin').text)
                # xmax = int(bbox.find('xmax').text)
                # ymax = int(bbox.find('ymax').text)
                # # 绘制边界框
                # color = (255, 255, 255)  # bai色边界框
                # font_color = (0, 0, 0)  # hei色边界框
                # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
                #
                # cv2.rectangle(image, (xmin, ymax), (xmin + len(class_name)*30 +10, ymax + 40), color, -1)
                # # cv2.putText(image, class_name, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
                # image = cv2ImgAddText(image,class_name, xmin+5, ymax, font_color,textSize=28)

            # output_image_path = os.path.join(detect_path, basename)
            # cv2.imwrite(output_image_path, image)


if __name__ == '__main__':
    detect_path = "/media/kevin/B044E2E97FC50881/Prohibited/yolov5_1/runs/detect/exp"
    gt_folder_path = "/media/kevin/B044E2E97FC50881/Prohibited/dataset/renameTest-30/Annotations/"
    draw_gt_multi(detect_path, gt_folder_path)
