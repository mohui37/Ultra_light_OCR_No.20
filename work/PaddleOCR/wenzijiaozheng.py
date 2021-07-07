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
def VThin(image, array):
    h,w= image.shape[:2]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j-1] + image[i,j] + image[i, j+1] if 0<j<w-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if-1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k, j-1+l] == 255:
                                a[k*3 + l] = 1
                    sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    h, w = image.shape[:2]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1, j] + image[i, j] + image[i+1, j] if 0<i<h-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k, j-1+l] == 255:
                                a[k*3 + l] = 1
                    sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def Xihua(binary, array, num=10):
    iXihua = binary.copy()
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua


filepath=os.listdir('test')
print(len(filepath),filepath[0])
num=0
for i in range(len(filepath)):
    path='test/'+filepath[i]
    #if i&2000==0:print(i)
    print(path)
    img=cv2.imread(path)
    #if entropy(img)<3.7 and int(filepath[i][6:-4])<5000:
    #    num+=1
    #if np.mean(img)<128:
    #    img=255-img
    #if  entropy(img)<=t :
    #    img2 = cv2.normalize(img,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    #    cv2.imwrite('/root/ppocr_mh/data/A榜测试数据集/TestAImages_clear3/'+filepath[i],img2)
    #    cv2.imwrite('/root/ppocr_mh/data/A榜测试数据集/TestAImages_mohu3/'+filepath[i],img)
    if np.mean(img)<=64:
        img2 = cv2.normalize(img,dst=None,alpha=355,beta=0,norm_type=cv2.NORM_MINMAX)
    else:
        img2=img
    #img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    #kernel = np.ones((5,5),np.uint8)
    #img2 = cv2.dilate(img2,kernel,iterations = 1)
    #img2 = Xihua(img2,array)
    w=img2.shape[1]
    h=img2.shape[0]
    if img2.shape[1]>=img2.shape[0]:
        n=100/img2.shape[0]
        img2 =cv2.resize(img2,(int(img2.shape[1]*n),int(img2.shape[0]*n)))
    if img2.shape[1]<img2.shape[0]:
        n=100/img2.shape[1]
        img2 =cv2.resize(img2,(int(img2.shape[1]*n),int(img2.shape[0]*n)))
    if np.mean(img2)<=128:
        img2=255-img2
    kernel = np.ones((3,3),np.uint8)
    #img2 = cv2.dilate(img2,kernel,iterations = 1)
    img2 = cv2.erode(img2,kernel,iterations = 3)
    #img2 = cv2.resize(img2,(w,h))

    cv2.imwrite('test_test/'+filepath[i],img2)
print(num)
