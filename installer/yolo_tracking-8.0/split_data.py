import os
import shutil

data_path = "D:\\Codes\\机场物流项目代码\\trunk_images"

files = os.listdir(data_path)
for index in range(0,len(files)):
    file_name = "frame_"+str(index)
    if os.path.exists(os.path.join(data_path,file_name+".jpg")) and os.path.exists(os.path.join(data_path,file_name+".txt")):
        shutil.copy(os.path.join(data_path,file_name+".jpg"),os.path.join(data_path,"images"))
        shutil.copy(os.path.join(data_path, file_name + ".txt"), os.path.join(data_path, "labels"))



