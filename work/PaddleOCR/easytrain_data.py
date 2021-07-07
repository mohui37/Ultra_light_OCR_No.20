import os 
import numpy
import random
with open("/root/ppocr_mh/data/训练数据集/LabelTrain.txt", "r") as f:  # 打开文件
    data = f.readlines()


print(len(data),data[0])
easy_data_5_val=[]
easy_data_15_val=[]
easy_data_5_train=[]
easy_data_15_train=[]
max_len=0
for i in range(len(data)):
    print(i)
    print(data[i][17:])
    if len(data[i][17:])<5:
        #print(data[i].split()[17:])
        if random.random()<0.8:
            easy_data_5_train.append(data[i])
        else:
            easy_data_5_val.append(data[i])
    if len(data[i][17:])<15:
        if random.random()<0.8:
            easy_data_15_train.append(data[i])
        else:
            easy_data_15_val.append(data[i])
    if len(data[i][17:])>max_len:
        max_len=len(data[i][17:])

print(max_len)
with open("easy_data_5_train.txt","w") as f:
    for i in range(len(easy_data_5_train)):
        f.write(easy_data_5_train[i])
with open("easy_data_15_train.txt","w") as f:
    for i in range(len(easy_data_15_train)):
        f.write(easy_data_15_train[i])
with open("easy_data_5_val.txt","w") as f:
    for i in range(len(easy_data_5_val)):
        f.write(easy_data_5_val[i])
with open("easy_data_15_val.txt","w") as f:
    for i in range(len(easy_data_15_val)):
        f.write(easy_data_15_val[i]) 