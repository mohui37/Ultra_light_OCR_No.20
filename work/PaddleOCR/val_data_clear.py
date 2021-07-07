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

def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
    	for y in range(1, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)+((int(img[x,y+1]-int(img[x,y])))**2)
    return out

def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''

    img= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out

def random_crop(image, min_ratio=0.6, max_ratio=1.0):

    h, w = image.shape[:2]
    
    ratio = random.random()
    
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    
    new_h = int(h*scale)    
    new_w = int(w*scale)
    
    y = np.random.randint(0, h - new_h)    
    x = np.random.randint(0, w - new_w)
    
    image = image[y:y+new_h, x:x+new_w, :]
    
    return image

def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img,cv2.CV_64F).var()

def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
    	for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

with open("/root/ppocr_mh/data/训练数据集/LabelTrain.txt", "r") as f:  # 打开文件
    data = f.readlines()

def jiacu(img):
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
    return img2


print(len(data),data[0])
wenben_jiucuo_5=[]
wenben_jiucuo_5_val=[]
easy_data_15_val=[]
easy_data_5_train=[]
easy_data_15_train=[]
max_len=0
num=0
for i in range(len(data)):
    print(i)
    print(data[i][:16],data[i][6:12])
    img=cv2.imread('/root/ppocr_mh/data/训练数据集/TrainImages/'+data[i][:16])
    img2=img
    t=3.0
    if  entropy(img)>=t and int(data[i][6:12])<50000:
        if random.random()<0.9:
            wenben_jiucuo_5.append(data[i])
        else:
            wenben_jiucuo_5_val.append(data[i])    
    if   int(data[i][6:12])>=50000:
        if random.random()<0.9:
            wenben_jiucuo_5.append(data[i])
        else:
            wenben_jiucuo_5_val.append(data[i])  
    if  entropy(img)<t and  int(data[i][6:12])<50000:
        num+=1
    
    #if  int(data[i][6:12])>=50000:
    #    img2 = jiacu(img)
    if  int(data[i][6:12])<50000 and  ((np.max(img)-np.mean(img))<=32 or np.mean(img)<56 or np.mean(img)>200):
        #img2= cv2.convertScaleAbs(img2,alpha=1.5,beta=0)
        img2= cv2.normalize(img2,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    #if np.mean(img2)>128:
    #    img2=255-img2
    cv2.imwrite('/root/ppocr_mh/data/训练数据集/TrainImages_clear3_0602/'+data[i][:16],img2)
    
        #cv2.imwrite('/root/ppocr_mh/data/训练数据集/TrainImages_clear3_origin/'+data[i][:16],img)
print(num)

with open("train_clear3_0602_train.txt","w") as f:
    for i in range(len(wenben_jiucuo_5)):
        f.write(wenben_jiucuo_5[i])
with open("train_clear3_0602_val.txt","w") as f:
    for i in range(len(wenben_jiucuo_5_val)):
        f.write(wenben_jiucuo_5_val[i])

'''
with open("train_clear_0527_0-50000_val.txt","w") as f:
    for i in range(len(wenben_jiucuo_5_val)):
        f.write(wenben_jiucuo_5_val[i])
with open("train_clear_0527_50000-100000_val.txt","w") as f:
    for i in range(len(easy_data_15_val)):
        f.write(easy_data_15_val[i])
'''

filepath=os.listdir('/root/ppocr_mh/data/A榜测试数据集/TestAImages')
print(len(filepath),filepath[0])
num=0
for i in range(len(filepath)):
    path='/root/ppocr_mh/data/A榜测试数据集/TestAImages/'+filepath[i]
    #if i&2000==0:print(i)
    print(path)
    img2=cv2.imread(path)
    #if entropy(img)<3.7 and int(filepath[i][6:-4])<5000:
    #    num+=1
    #if np.mean(img)<128:
    #    img=255-img
    #if  entropy(img)<=t :
    #    img2 = cv2.normalize(img,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    #    cv2.imwrite('/root/ppocr_mh/data/A榜测试数据集/TestAImages_clear3/'+filepath[i],img2)
    #    cv2.imwrite('/root/ppocr_mh/data/A榜测试数据集/TestAImages_mohu3/'+filepath[i],img)
    print(filepath[i][6:-4])
    #if  int(filepath[i][6:-4])>=5000:
    #    img2 = jiacu(img2)
    if  int(filepath[i][6:-4])<5000 and  ((np.max(img)-np.mean(img))<=64 or np.mean(img)<56 or np.mean(img)>200):
        #img2= cv2.convertScaleAbs(img2,alpha=1.5,beta=0)
        img2= cv2.normalize(img2,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    
    cv2.imwrite('/root/ppocr_mh/data/A榜测试数据集/TestAImages_clear3_0602/'+filepath[i],img2)
print(num)
