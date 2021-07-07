import os 
import numpy
import random
with open("/root/ppocr_mh/data/训练数据集/LabelTrain.txt", "r") as f:  # 打开文件
    data = f.readlines()


print(len(data),data[0])
wenben_jiucuo_5=[]
easy_data_15_val=[]
easy_data_5_train=[]
easy_data_15_train=[]
max_len=0
for i in range(len(data)):
    print(i)
    print(data[i][17:])
    if len(data[i][17:])>2:
        wenben_jiucuo_5.append(data[i][17:])


with open("wenben_jiucuo_5.txt","w") as f:
    for i in range(len(wenben_jiucuo_5)):
        f.write(wenben_jiucuo_5[i])
'''with open("easy_data_15_train.txt","w") as f:
    for i in range(len(easy_data_15_train)):
        f.write(easy_data_15_train[i])
with open("easy_data_5_val.txt","w") as f:
    for i in range(len(easy_data_5_val)):
        f.write(easy_data_5_val[i])
with open("easy_data_15_val.txt","w") as f:
    for i in range(len(easy_data_15_val)):
        f.write(easy_data_15_val[i]) '''