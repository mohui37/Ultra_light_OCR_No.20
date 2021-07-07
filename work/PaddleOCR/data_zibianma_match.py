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


with open("/root/ppocr_mh/data/训练数据集/LabelTrain.txt", "r") as f:  # 打开文件
    data = f.readlines()
with open("/root/ppocr_mh/work/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt", "r") as f:  # 打开文件
    label = f.readlines()
print(len(label),label[0])
tongji_0_5=np.zeros((len(label)))
tongji_5_10=np.zeros((len(label)))

for i in range(int(len(data))):
    print(i)
    #print(data[i][17:])
    temp=data[i][17:]
    for j in range(len(temp)):
        for k in range(len(label)):
            
            if temp[j]==label[k][0]:
                #print(temp[j])
                #print(label[k][0])
                if i<50000:
                    tongji_0_5[k]+=1
                if i>=50000:
                    tongji_5_10[k]+=1
print(tongji_0_5.shape)
print(tongji_5_10.shape)
t=tongji_0_5*tongji_5_10
print(t.shape)

new_label=[]
num=0
for i in range(len(tongji_0_5)):
    if tongji_0_5[i]>0:
        new_label.append(label[i])
        num+=1
print(num)
num=0
for i in range(len(tongji_5_10)):
    if tongji_5_10[i]>0:
        new_label.append(label[i])
        num+=1
print(num)
num=0
for i in range(len(t)):
    if t[i]>0:
        new_label.append(label[i])
        num+=1
print(num)

with open("/root/ppocr_mh/work/PaddleOCR/ppocr/utils/ppocr_keys_v3.txt","w") as f:
    #f.write(label[0])
    for i in range(len(new_label)):
        f.write(new_label[i])

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4))

# 垂直柱状图
ax1 = fig.add_subplot(111)
ax1.set_title('图1 垂直柱状图')
ax1.bar(x=range(len(label)), height=t)
fig.savefig(r"match.png", transparent=True)



'''
from radical import Radical

if __name__ == '__main__':
    radical = Radical()

    # 如果需要查找的字在字典中，则直接返回其偏旁部首
    print radical.get_radical('好')

    # 本地词典查不到，则从百度汉语中查找
    print radical.get_radical('淥')

    # 可通过下面操作保存新加入的字
    # radical.save()
    '''