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

'''
with open("/root/ppocr_mh/data/训练数据集/LabelTrain.txt", "r") as f:  # 打开文件
    data = f.readlines()
with open("/root/ppocr_mh/work/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt", "r") as f:  # 打开文件
    label = f.readlines()
print(len(label),label[0])
tongji=np.zeros((len(label)))
for i in range(int(len(data))):
    print(i)
    #print(data[i][17:])
    temp=data[i][17:]
    for j in range(len(temp)):
        for k in range(len(label)):
            
            if temp[j]==label[k][0]:
                #print(temp[j])
                #print(label[k][0])
                tongji[k]+=1

new_label=[]
num=0
for i in range(len(tongji)):
    if tongji[i]>0:
        new_label.append(label[i])
        num+=1
print(num)

with open("/root/ppocr_mh/work/PaddleOCR/ppocr/utils/ppocr_keys_v2.txt","w") as f:
    #f.write(label[0])
    for i in range(len(new_label)):
        f.write(new_label[i])

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4))

# 垂直柱状图
ax1 = fig.add_subplot(111)
ax1.set_title('图1 垂直柱状图')
ax1.bar(x=range(len(label)), height=tongji)
fig.savefig(r"bar_img.png", transparent=True) # 
'''

from PIL import Image, ImageDraw, ImageFont
#生成汉字图片，计算相似重构label等级
with open("/root/ppocr_mh/work/PaddleOCR/ppocr/utils/ppocr_keys_v2.txt", "r") as f:  # 打开文件
    label = f.readlines()

for k in range(len(label)):
    print(label[k][0])
    img=np.zeros((48,48,3))*255
    img=np.uint8(img)
    #print(img.shape)
    #img=img.astype('float')
    img = Image.fromarray(np.asarray(img))
    font = ImageFont.truetype(
        "simsun.ttc", 38, encoding="utf-8")
    # 图片对象、文本、位置像素、字体、字体大小、颜色、字体粗细
    draw = ImageDraw.Draw(img)
    str2 = label[k][0]#.decode('utf8')
    #str2=str2.encode('utf-8').decode('utf-8')
    draw.text((5,5), str2, font=font,fillColor = (255,255,255))
    #img = cv2.putText(img, label[k][0], (15, 15), font, 0.5, (0, 0, 0), 1)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    cv2.imwrite('/root/ppocr_mh/work/PaddleOCR/ppocr/utils/hanzi/'+str(k)+'.jpg',img)