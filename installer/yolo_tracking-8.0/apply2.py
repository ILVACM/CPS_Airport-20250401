# 导入需要的库
import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import ImageFont,ImageDraw,Image

from trackers.multi_tracker_zoo import create_tracker
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

# 导入letterbox
from yolov5.utils.augmentations import letterbox

# 初始化目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到环境变量中（程序结束后删除）
WEIGHTS = ROOT / 'weights'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Detect:
    def __init__(self, weights=ROOT / 'yolov5/best.pt', data=ROOT / 'yolov5/data/prohibited.yaml',reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='bytetrack',half=False):
        self.cls_names = ["枪支", "电击器", "警棍", "手铐", "管制刀具", "鞭炮", "电子烟花", "礼花", "烟花", "烟饼",
                          "仙女棒",
                          "压缩罐", "高锰酸钾", "子弹", "空包弹", "霰弹", "烟雾弹", "塑料打火机", "金属打火机",
                          "镁棒点火器",
                          "防风火柴", "块状电池", "节状电池", "纽扣电池", "蓄电池", "蓝牙耳机"]


        self.font = ImageFont.truetype('simhei.ttf', 20)
        self.imgsz = (640, 640)  # 输入图片的大小 默认640(pixels)
        self.conf_thres = 0.25  # object置信度阈值 默认0.25  用在nms中
        self.iou_thres = 0.45  # 做nms的iou阈值 默认0.45   用在nms中
        self.max_det = 1000  # 每张图片最多的目标数量  用在nms中
        self.classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
        self.agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
        self.augment = False  # 预测是否也要采用数据增强 TTA 默认False
        self.visualize = False  # 特征图可视化 默认FALSE
        self.half = False  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
        dnn = False  # 使用OpenCV DNN进行ONNX推理
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride)  # 检查图片尺寸
        self.half &= (self.model.pt or self.model.jit or self.model.onnx or self.model.engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.model.pt or self.model.jit:
            self.model.model.half() if self.half else self.model.model.float()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        self.model.eval()
        self.count = 0
        #
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # print(weights)
        self.video = cv2.VideoWriter('test.mp4', fourcc, 10.0, (2160,1920))
        self.start_time = time.time()
        tracker_list = []
        nr_sources = 2
        for i in range(nr_sources):
            tracker = create_tracker(tracking_method, reid_weights, self.device, self.half)
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        self.outputs = [None] * nr_sources
        self.tracker_list = tracker_list
        self.curr_frames, self.prev_frames = [None] * nr_sources, [None] * nr_sources
        self.count = 0

    def detect(self, img,input_view,threshold):
        # print(img.shape)
        t1 = time_sync()
        # Run inference
        # 开始预测
        dt, seen = [0.0, 0.0, 0.0], 0
        # 对图片进行处理

        im0 = img.copy()
        # cv2.imwrite('./images2/' + str(self.count) + '.jpg', im0)
        self.count += 1
        scale_shape = im0.shape

        # Padded resize
        if input_view==2:
            scale_shape = im0[0].shape
            im_up = letterbox(im0[0], self.imgsz, self.model.stride, auto=self.model.pt)[0]
            im_down = letterbox(im0[1], self.imgsz, self.model.stride, auto=self.model.pt)[0]

            # Convert
            im = np.stack([im_up,im_down],axis=0)
            im = im.transpose((0,3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
        else:
            im = letterbox(im0, self.imgsz, self.model.stride, auto=self.model.pt)[0]
            # Convert
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # 预测

        pred = self.model(im, augment=self.augment, visualize=self.visualize)


        # NMS
        pred = non_max_suppression(pred, threshold, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        t3 = time_sync()
        dt[1] = t3 - t2


        # Process predictions
        labels_r = []
        annotator_up = Annotator(im0, line_width=3, example="0")
        # annotator_down = Annotator(im0[1], line_width=3, example=str(self.cls_names))

        for i, det in enumerate(reversed(pred)):  # per image 每张图片
            if input_view == 2:
                self.curr_frames[i] = im0[i]
            else:
                self.curr_frames[i] = im0

            seen += 1
            # print("len(det):",len(det))
            labels = []
            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], scale_shape).round()
                # print(self.curr_frames[i].shape)
                self.outputs[i] = self.tracker_list[i].update(det.cpu(), self.curr_frames[i])
                # print("len(self.outputs[i]):",len(self.outputs[i]))
                # Write results
                # 写入结果

                # for *xyxy, conf, cls in reversed(det):
                #
                #     cls = int(cls)
                #     xyxy = [int(i) for i in xyxy]
                #     if i == 0:
                #         annotator_up.box_label(xyxy, self.cls_names[cls], color=colors(cls, True))
                #     else:
                #         annotator_down.box_label(xyxy, self.cls_names[cls], color=colors(cls, True))
                #     labels.append({ 'label': self.cls_names[cls], 'xyxy': xyxy, 'conf': float(conf)})


                if len(self.outputs[i]) > 0:

                    for j, (output, conf) in enumerate(zip(self.outputs[i], det[:, 4])):
                        xyxy = output[0:4]
                        id = int(output[4])
                        cls = int(output[5])
                        xyxy = [int(i) for i in xyxy]
                        labels.append({'id':id,'label': cls, 'xyxy': xyxy, 'conf': float(conf)})
                        annotator_up.box_label(xyxy, cls, color=colors(cls, True))
                # if i == 0:
                #     cv2.imwrite(os.path.join("./images/", str(self.count)+"_up.jpg"),annotator_up.result())
                # else:
                #     cv2.imwrite(os.path.join("./images/", str(self.count) + "_down.jpg"), annotator_down.result())
                # self.count += 1

                self.prev_frames[i] = self.curr_frames[i]
            labels_r.append(labels)

        dt[2] += time_sync() - t3

        # print(dt)
        # labels_r.reverse()
        return  annotator_up.result(),labels_r


    def detect_upload(self, img):
        # Run inference
        # 开始预测
        label1 = []
        xy_list = []
        dt, seen = [0.0, 0.0, 0.0], 0
        # 对图片进行处理
        im0 = img.copy()
        annotator = Annotator(im0, line_width=3, example=str(self.model.names))
        # Padded resize
        im = letterbox(im0, self.imgsz, self.model.stride, auto=self.model.pt)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # 预测
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image 每张图片
            seen += 1

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # print(det)
                # Write results
                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.cls_names[c]} {conf:.2f}'
                    im0=draw_china(im0,xyxy,3,self.font, label, color=colors(c, True))

                    xy_list = []

                    for xy in xyxy:
                        xy_list.append(int(xy))
                    #
                    # imc = save_one_box(xyxy, img.copy(), BGR=True, save=False)
                    if c < len(self.cls_names):
                        label = self.cls_names[cls.to(int)]

                    label1.append({'label': label, 'xy': xy_list, 'conf': round(float(conf), 2)})

        return im0, label1

def draw_china(img, box, line_width, font, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    sf = line_width//3
    tf = max(line_width - 1, 1)
    cv2.rectangle(img, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)

    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w//2, p1[1] - h - 3 if outside else p1[1] + h + 3

        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled

        # 将NumPy数组转换为PIL图像
        pil_image = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_image)

        # 绘制中文文本
        draw.text((p1[0], p1[1] - 2 -h if outside else p1[1] + h + 2), label, font=font, fill=txt_color)

        # 将PIL图像转回NumPy数组
        image_with_chinese = np.array(pil_image)
        img = cv2.cvtColor(image_with_chinese,cv2.COLOR_RGB2BGR)
    return img

if __name__ == '__main__':
    weights = ROOT / 'runs/train/aug4-ppppplus-1228/weights/best.pt'  # 权重文件地址   .pt文件
    data = ROOT / 'data/prohibited.yaml'  # 标签文件地址   .yaml文件
    path = '/media/airport/E9E972F569CF8A7F/dataset/renameTest-30/rename4-t30/0e4de0447cd04a96b81b12f41aa50454.jpg'  # 检测图片的文件夹路径
    Det = Detect(weights=weights, data=data)
    result = Det.detcet(path)
    print(result)