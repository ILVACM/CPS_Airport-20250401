import base64
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator, colors


from flask import Flask

from apply2 import Detect

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
detect = Detect()
res, frame = cap.read()


i = 2
while res:

    start_time = time.time()
    # print(frame.shape)
    frame = cv2.resize(frame, dsize=(frame.shape[0] // 2, frame.shape[1] // 2))
    frame = frame[:,:900]
    input_view = 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("./images/" + str(data_test.i) + "_gray.jpg", gray)
    # 3. 二值化
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("./images/" + str(data_test.i) + "_binary.jpg", binary)
    cv2.imshow("image", binary)
    cv2.waitKey(500)
    # 4. 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 5. 获取外接矩形并绘制

    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 检查裁剪区域是否超出图像边界
        if w >= 50 and h >= 50 :
            print(rect)
            if x + w > frame.shape[1] or y + h > frame.shape[0]:
                # 调整裁剪参数以不超出图像边界
                w_adjusted = min(w, frame.shape[1] - x)
                h_adjusted = min(h, frame.shape[0] - y)
                # 创建一个新的空白图像，大小与调整后的裁剪区域相同，并填充为黑色（或其他颜色）
                new_image = np.zeros((h, w, 3), dtype=np.uint8)  # 黑色背景
                # 将原始图像中的裁剪区域复制到新图像
                new_image[:h_adjusted, :w_adjusted] = frame[y:y + h_adjusted, x:x + w_adjusted]
            else:
                # 裁剪区域没有超出图像边界，直接进行裁剪
                new_image = frame[y:y + h, x:x + w]
            cv2.imshow("new_image", new_image)
            cv2.waitKey(500)
    # if input_view == 2:
    #     mid = frame.shape[0] // 2
    #     frame1 = frame[0:mid]
    #     frame2 = frame[mid:]
    #     # cv2.imshow("up",frame1)
    #     # cv2.imshow("down", frame2)
    #     # cv2.waitKey(2)
    #     frame = np.stack([frame1, frame2], axis=0)
    # label = detect.detect(frame,input_view,0.25)

    # print(label)
    # img = cv2.resize(img, dsize=(img.shape[0] // 2, img.shape[1] // 2))
    print('spend time:', (time.time() - start_time))
    # cv2.imshow('result',frame)
    # cv2.waitKey(2)
    # print('labels:',len(label))
    res, frame = cap.read()


    #
    # if i == 2:
    # # i = 0
    # else:
    #     annotator = Annotator(frame, line_width=3,example=str(cls_names))
    #     for j, (output, conf) in enumerate(zip(outputs, det[:, 4])):
    #         bboxes = output[0:4]
    #         id = output[4]
    #         cls = output[5]
    #         c = int(cls)  # integer class
    #         id = int(id)  # integer id
    #         label = f'{id} {cls_names[c]} {conf:.2f}'
    #
    #         annotator.box_label(list(bboxes), label, color=colors(c, True))
    #     img = annotator.result()
    #     img = cv2.resize(img, dsize=(img.shape[0] // 2, img.shape[1] // 2))
    #     cv2.imshow('result', img)

    # i += 1