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

from scipy.spatial.distance import pdist
import time

import os
def euclidean(image1, image2):
    '''欧氏距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'euclidean')[0]


def manhattan(image1, image2):
    '''曼哈顿距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'cityblock')[0]


def chebyshev(image1, image2):
    '''切比雪夫距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'chebyshev')[0]


def cosine(image1, image2):
    '''余弦距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'cosine')[0]


def pearson(image1, image2):
    '''皮尔逊相关系数'''
    X = np.vstack([image1, image2])
    return np.corrcoef(X)[0][1]


def hamming(image1, image2):
    '''汉明距离'''
    return np.shape(np.nonzero(image1 - image2)[0])[0]


def jaccard(image1, image2):
    '''杰卡德距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'jaccard')[0]


def braycurtis(image1, image2):
    '''布雷柯蒂斯距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'braycurtis')[0]


def mahalanobis(image1, image2):
    '''马氏距离'''
    X = np.vstack([image1, image2])
    XT = X.T
    return pdist(XT, 'mahalanobis')


def jensenshannon(image1, image2):
    '''JS散度'''
    X = np.vstack([image1, image2])
    return pdist(X, 'jensenshannon')[0]


def image_match(image1, image2):
    '''image-match匹配库'''
    try:
        from image_match.goldberg import ImageSignature
    except:
        return -1
    image1 = ImageSignature().generate_signature(image1)
    image2 = ImageSignature().generate_signature(image2)
    return ImageSignature.normalized_distance(image1, image2)



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