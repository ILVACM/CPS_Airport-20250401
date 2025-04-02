import argparse
import base64
import json
import multiprocessing
import os
import threading
import time
import traceback

import cv2
import numpy as np
import torch

from apply3 import Detect
from flask import Flask, request

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# cap = cv2.VideoCapture('./测试视频.mp4')

# res, frame = cap.read()


app = Flask(__name__)


#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('test.mp4', fourcc, 10.0,
#                       (960,1080))


# 视图函数（路由）
@app.route('/detect', methods=['POST'])
def detect_image():
    # print(request.data)
    # dt = [0.0, 0.0, 0.0]
    # t1 = time.time()
    json_data = json.loads(request.data)
    decode_queue.put(json_data)
    data = encoded_queue.get()
    # print('spend time:', time.time() - t1)
    return json.dumps(data)











def detect_method(decode_queue, encoded_queue,args):
    detector = Detect(yolov8_weights=args.weights1,weights=args.weights2)
    while True:
        try:
            data = decode_queue.get()
            img_str = data["image"]
            # input_view = data["input_view"]
            threshold = data["threshold"]
            isDetect = data["isDetect"]
            str_decode = base64.b64decode(img_str)
            nparr = np.frombuffer(str_decode, np.uint8)
            # start_time = time.time()
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            track_result, labels = detector.detect(frame, threshold, isDetect)
            encoded_queue.put({"label": labels, "track_result": track_result, "isDetect": isDetect})

        except Exception as e:
            traceback.print_exc()
            encoded_queue.put({"label": {}, "track_result": {}, "isDetect": 0})


# def encode_method(encode_queue, encoded_queue):
#     while True:
#         if encode_queue.empty():
#             continue
#         print("encode_method")
#         img, label, det = encode_queue.get()
#         img_str = cv2.imencode('.jpg', img)[1]  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
#         b64_code = base64.b64encode(img_str).decode('utf-8')  # 编码成base64
#         data = {'image': b64_code, 'label': label}
#         encoded_queue.put(data)


# 启动实施（只在当前模块运行）
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program description')

    parser.add_argument('-w1', '--weights1', help='检测行李箱的权重', default='weights/trunk_best.pt')
    parser.add_argument('-w2', '--weights2', help='检测违禁品的权重', default='weights/last0930.pt')
    parser.add_argument('-p', '--port', help='服务端口', default=8000)
    # 解析命令行参数
    args = parser.parse_args()
    decode_queue = multiprocessing.Queue(maxsize=1)
    encoded_queue = multiprocessing.Queue(maxsize=1)
    detect_thread = multiprocessing.Process(target=detect_method, args=(decode_queue, encoded_queue,args))
    detect_thread.start()
    app.run(port=args.port)

