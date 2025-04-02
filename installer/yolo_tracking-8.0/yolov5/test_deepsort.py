import cv2
import torch
from deepsort import DeepSORT
from yolov5 import YOLOv5

# 初始化YOLOv5目标检测模型
model = YOLOv5("yolov5s.pt", device="cpu")  # 使用YOLOv5的小模型，并在CPU上运行

# 初始化DeepSORT跟踪器
deep_sort = DeepSORT(max_dist=0.2, min_confidence=0.5, nms_max_overlap=0.5)

# 加载视频文件或打开摄像头
cap = cv2.VideoCapture("video.mp4")  # 替换为你的视频文件路径或摄像头ID（如0）

frame_rate = 30.0  # 假设的视频帧率，或根据实际情况获取
codec = cv2.VideoWriter_fourcc(*'mp4v')  # 输出视频的编解码器
out = cv2.VideoWriter('output.mp4', codec, frame_rate, (int(cap.get(3)), int(cap.get(4))))  # 输出视频文件

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

        # 使用YOLOv5进行目标检测
    results = model(frame)
    xyxys = results.xyxy[0]  # 获取边界框坐标
    confidences = results.conf[0]  # 获取置信度分数
    class_ids = results.cls[0]  # 获取类别ID

    # 提取检测结果和分数
    detections = [Detection(bbox, confidence, class_id) for bbox, confidence, class_id in
                  zip(xyxys, confidences, class_ids) if confidence > 0.4]

    # 使用DeepSORT进行多目标跟踪
    tracks = deep_sort.update(detections, frame)

    # 绘制跟踪结果
    for track in tracks:
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 0), 2)

        # 显示结果或保存视频帧
    cv2.imshow("Multi-Object Tracking", frame)
    out.write(frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 释放资源并关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()