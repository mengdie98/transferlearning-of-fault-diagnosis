# import os

# # 设置源文件夹路径和目标文件夹路径
# src_folder = r"E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\baseline\with_box"
# dest_folder = r"E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\datapic\with_box"

# # 遍历源文件夹中的所有文件
# for filename in os.listdir(src_folder):
#     # 排除非CSV文件
#     if not filename.endswith(".csv"):
#         continue
    
#     # 解析文件名中的转速、载荷和故障种类
#     parts = filename[:-4].split("_")
#     speed = parts[-3]
#     load = parts[-2]
#     fault_type = "_".join(parts[:-3])
    
#     # 构造目标文件夹路径
#     dest_path = os.path.join(dest_folder, speed, load, fault_type, filename)
    
#     # 如果目标文件夹不存在，则创建它
#     if not os.path.exists(os.path.dirname(dest_path)):
#         os.makedirs(os.path.dirname(dest_path))
    
#     # 将文件移动到目标文件夹
#     src_path = os.path.join(src_folder, filename)
#     # os.rename(src_path, dest_path)

import os
import random
import pathlib

# 定义路径和目标文件夹
path = r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\datapic\with_box\1500\0'  # 原始文件夹路径
path2 = r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\datapic\with_box\1500\0t'  # 目标文件夹路径

# 遍历每个子文件夹
for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        # 创建目标文件夹
        target_dir = os.path.join(path2, subdir)
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

        # 获取子文件夹中的所有图片
        images = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        num_images = len(images)

        # 随机选择20%的图片，并将它们移动到目标文件夹中
        num_samples = int(num_images * 0.2)
        sample_images = random.sample(images, num_samples)
        for image in sample_images:
            src_path = os.path.join(subdir_path, image)
            dst_path = os.path.join(target_dir, image)
            os.rename(src_path, dst_path)



# import os
# import shutil

# # 设置源文件夹路径
# src_folder = "path/to/source/folder"

# # 遍历源文件夹中的所有文件
# for filename in os.listdir(src_folder):
#     # 排除非CSV文件
#     if not filename.endswith(".csv"):
#         continue
    
#     # 解析文件名中的转速、载荷和故障种类
#     parts = filename[:-4].split("_")
#     speed = parts[-3]
#     load = parts[-2]
#     fault_type = "_".join(parts[:-3])
    
#     # 构造目标文件夹路径
#     dest_folder = os.path.join("path/to/destination/folder", speed, load, fault_type)
    
#     # 如果目标文件夹不存在，则创建它
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)
    
#     # 将文件移动到目标文件夹
#     src_path = os.path.join(src_folder, filename)
#     dest_path = os.path.join(dest_folder, filename)
#     shutil.move(src_path, dest_path)
