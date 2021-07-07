import os 
import numpy
import random
import cv2
import torch
import torchvision
import cv2
from PIL import Image
import numpy as np
import os
#print(torch.cuda.get_device_name(0))

import math

 
import time

import os


with open("/root/ppocr_mh/data/B榜测试数据集/predicts_chinese_common_v2.0.txt", "r") as f:  # 打开文件
    data = f.readlines()
new_label=[]

for i in range(int(len(data))):
    print(i)
    temp=data[i].split()
    print(temp[0][31:]+'  '+temp[1])
   
    new_label.append(temp[0][31:]+'  '+temp[1]+'\n')


with open("/root/ppocr_mh/data/B榜测试数据集/label.txt","w") as f:
    #f.write(label[0])
    for i in range(len(new_label)):
        f.write(new_label[i])
