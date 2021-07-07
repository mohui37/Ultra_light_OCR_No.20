import os 
import numpy
import random
with open("/root/ppocr_mh/work/PaddleOCR/output/rec/predicts_chinese_lite_v2.0_1.0_50000_100000_2.txt", "r") as f:  # 打开文件
    data = f.readlines()


print(len(data),data[0])
wenben_jiucuo_5=[]
easy_data_15_val=[]
easy_data_5_train=[]
easy_data_15_train=[]
max_len=0
num=0
for i in range(len(data)):
    '''if data[i].split()[-1]=='nan':
        print('nan',i)
        continue'''
    #print(i)
    s = len(data[i].split()[0])
    e = len(data[i].split()[-1])
    #print(len(data[i].split()[0]),len(data[i].split()[-1]))
    #print(data[i][s+1:-e-2])
    print(data[i][-e-1:])
    #print(data[i][-1:])

    '''if float(data[i][-e:-1])<0.0 :#and len(data[i][s+1:-e-2])>5:
        num+=1
        print(i)
        print(data[i][s+1:-e-2])'''
    print(data[i][s-10:s-4])
    
    if int(data[i][s-10:s-4])<5000:
        wenben_jiucuo_5.append(data[i][:s+1]+'tttttt '+'0.000000\n')
    if int(data[i][s-10:s-4])>=5000:
        wenben_jiucuo_5.append(data[i])
print(num)
        


with open("0_5000_100000_0530.txt","w") as f:
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