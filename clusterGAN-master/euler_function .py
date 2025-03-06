# import os
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import io
# from quantimpy import morphology as mp
# from quantimpy import minkowski as mk
# import cv2

# # 用于计算欧拉数的函数 (Modified to use quantimpy)
# def calculate_euler_number(image_path):
#     # 读取图像
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # 二值化图像（将图像转为黑白）
#     _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

#     # 使用 quantimpy 计算欧拉数
#     # 生成形态学侵蚀图像
#     erosion_map = mp.erode_map(binary_image.astype(bool))

#     # 计算 Minkowski 函数度量
#     dist, area, length, euler = mk.functions_close(erosion_map)

#     # 假设只保存第一个距离点的欧拉值
#     euler_number = euler[0]  # 取得第一个值
#     return float(euler_number)

# # 存储欧拉值数据的列表
# synthetic_euler = []
# real_euler = []

# # 加载合成Berea砂岩图像的孔隙度数据
# synthetic_folder = r'./runs/my_dataset/20quan-c600s-1e-4-210epoch_z64_wass_bs4_test_run/images/0'
# synthetic_images = os.listdir(synthetic_folder)[:100]  # 只使用前100张图像
# i = 0
# for image_name in synthetic_images:
#     print(i)
#     image_path = os.path.join(synthetic_folder, image_name)
#     euler = calculate_euler_number(image_path)
#     synthetic_euler.append(euler)
#     i += 1

# # 加载真实图像的孔隙度数据
# real_folder = './runs/my_dataset/20quan-c600s-1e-4-210epoch_z64_wass_bs4_test_run/images/1'
# real_images = os.listdir(real_folder)[:100]  # 只使用前100张图像
# j = 0
# for image_name in real_images:
#     print('j=', j)
#     image_path = os.path.join(real_folder, image_name)
#     euler = calculate_euler_number(image_path)
#     real_euler.append(euler)
#     j += 1

# # 创建箱线图
# plt.figure(figsize=(9, 7))
# plt.boxplot([synthetic_euler, real_euler], labels=['class1', 'class2'])
# plt.title('Minkowski Euler')
# plt.ylabel('Euler')
# plt.savefig('欧拉值.png')
# plt.show()
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from quantimpy import morphology as mp
from quantimpy import minkowski as mk
import cv2

# 用于计算欧拉数的函数 (Modified to use quantimpy)
def calculate_euler_number(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二值化图像（将图像转为黑白）
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # 使用 quantimpy 计算欧拉数
    # 生成形态学侵蚀图像
    erosion_map = mp.erode_map(binary_image.astype(bool))

    # 计算 Minkowski 函数度量
    dist, area, length, euler = mk.functions_close(erosion_map)

    # 假设只保存第一个距离点的欧拉值
    euler_number = euler[0]  # 取得第一个值
    return float(euler_number)

# 存储欧拉值数据的列表
synthetic_euler = []
real_euler = []

# 加载合成Berea砂岩图像的孔隙度数据
synthetic_folder = r'./data-c-800s/3'
synthetic_images = os.listdir(synthetic_folder)[:100]  # 只使用前100张图像
i = 0
for image_name in synthetic_images:
    print(i)
    image_path = os.path.join(synthetic_folder, image_name)
    euler = calculate_euler_number(image_path)
    synthetic_euler.append(euler)
    i += 1

# 加载真实图像的孔隙度数据
real_folder = './data-c-800s/2'
real_images = os.listdir(real_folder)[:100]  # 只使用前100张图像
j = 0
for image_name in real_images:
    print('j=', j)
    image_path = os.path.join(real_folder, image_name)
    euler = calculate_euler_number(image_path)
    real_euler.append(euler)
    j += 1

# 创建箱线图并获取返回的字典对象
plt.figure(figsize=(9, 7))
boxplot = plt.boxplot([synthetic_euler, real_euler], labels=['class1', 'class2'])

# 获取箱线图的统计数据
synthetic_stats = boxplot['medians'][0].get_ydata()  # 获取合成图像的中位数
real_stats = boxplot['medians'][1].get_ydata()  # 获取真实图像的中位数

# 输出两个箱线图的上边缘线（Q3）和下边缘线（Q1）的值
print(f"Synthetic images boxplot Q1 (lower whisker) = {boxplot['whiskers'][0].get_ydata()[0]}, "
      f"Q3 (upper whisker) = {boxplot['whiskers'][1].get_ydata()[1]}")

print(f"Real images boxplot Q1 (lower whisker) = {boxplot['whiskers'][2].get_ydata()[0]}, "
      f"Q3 (upper whisker) = {boxplot['whiskers'][3].get_ydata()[1]}")

# 创建并保存箱线图
plt.title('Minkowski Euler')
plt.ylabel('Euler')
plt.savefig('欧拉值.png')
plt.show()
