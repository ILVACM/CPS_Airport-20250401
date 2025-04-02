import os.path

import cv2
import numpy as np

# 设置视频文件路径和帧率
video_path = 'D:\\BaiduNetdiskDownload\\20240115采集视频（违禁品）\\采集视频（违禁品）\\7号采集视频.mp4'
frame_rate = 30.0  # 假设视频的帧率为30fps
frame_interval = 50# 每5帧抽取一帧

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 读取第一帧并保存为上一帧
ret, prev_frame = cap.read()
if not ret:
    print("无法读取视频文件或视频已结束")
    exit()

# 初始化一个计数器来跟踪帧数
frame_count = 0
images_path = "./images/"
# 遍历视频的每一帧
frame_name = 197
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count == 0:
        frame_filename = f'frame_{frame_name}.jpg'
        cv2.imwrite(os.path.join(images_path, frame_filename), frame)
        frame_name += 1
        print(f"保存帧: {frame_filename}")
    # 判断是否达到了抽取帧的间隔
    if frame_count % frame_interval == 0:
        # 计算当前帧与上一帧的差异
        diff = cv2.absdiff(frame, prev_frame)

        # 使用某种阈值来判断两帧是否“相似”
        # 这里使用简单的平均值阈值作为示例
        mean_diff = np.mean(diff)
        print(mean_diff)
        # 如果差异大于某个阈值，则认为帧是不同的
        if mean_diff > 30:  # SOME_THRESHOLD 是你需要设定的阈值
            # 保存新帧为图像文件
            frame_filename = f'frame_{frame_name}.jpg'
            cv2.imwrite(os.path.join(images_path,frame_filename), frame)
            frame_name += 1
            print(f"保存帧: {frame_filename}")

            # 更新上一帧为当前帧
        prev_frame = frame.copy()
    frame_count += 1
print(frame_name)
    # 释放视频文件和窗口
cap.release()
cv2.destroyAllWindows()