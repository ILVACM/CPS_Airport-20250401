
import os


import cv2
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator, colors

from apply3 import Detect

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

cls_names = ["枪支", "电击器", "警棍", "手铐", "管制刀具", "鞭炮", "电子烟花", "礼花", "烟花", "烟饼",
                          "仙女棒",
                          "压缩罐", "高锰酸钾", "子弹", "空包弹", "霰弹", "烟雾弹", "塑料打火机", "金属打火机",
                          "镁棒点火器",
                          "防风火柴", "块状电池", "节状电池", "纽扣电池", "蓄电池", "蓝牙耳机"]


print(torch.cuda.is_available())

cap = cv2.VideoCapture('./测试视频.mp4')
# detect = Detect("../best.pt")
res, frame = cap.read()
detect = Detect()
# files = os.listdir("./images")
i = 0
last_labels = {}
while res:

    input_view = 1
    if input_view == 2:
        mid = frame.shape[0] // 2
        frame1 = frame[0:mid]
        frame2 = frame[mid:]
        frame = np.stack([frame1, frame2], axis=0)
    print(i%10)
    labels = {}
    if i % 10 == 0:
        track_result, labels = detect.detect(frame, 0.25,0)
        last_labels = labels
    else:
        track_result, labels = detect.detect(frame, 0.25,1)
        labels = last_labels

    annotator = Annotator(frame, line_width=3, example=str(cls_names))
    i += 1
    # print(labels)
    for id in track_result.keys():
        annotator.box_label(track_result[id], str(id) + "_trunk", color=colors(0, True))
        if id in labels.keys():
            for label_dict in labels[id]:
                xyxy = label_dict['xyxy'][:]
                xyxy[0] +=  track_result[id][0]
                xyxy[1] += track_result[id][1]
                xyxy[2] += track_result[id][0]
                xyxy[3] += track_result[id][1]
                print(xyxy)
                label = label_dict['label']
                cls = label_dict['cls']
                annotator.box_label(xyxy, label, color=colors(cls, True))

    result_image = annotator.result()
    result_image = cv2.resize(result_image,dsize=(result_image.shape[0]//2,result_image.shape[1]//2))

    cv2.imshow("result_image",result_image)
    cv2.waitKey(100)
    res, frame = cap.read()




