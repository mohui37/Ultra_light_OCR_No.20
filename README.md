# 轻量级文字识别技术创新大赛第20名方案

## 项目描述
简要描述项目

## 运行环境
2080ti单卡
cuda10.2
cudnn7.6


## 数据集的上传与下载（大文件）
注意：git lfs 要求 git >= 1.8.2

Linux

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs

git lfs install

Mac

安装HomeBrew /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

brew install git-lfs

git lfs install

## 数据集解压
下载后，将在data目录的A\B\训练测试集.zip，解压到data目录


## 项目结构
```
-|data
    |A
    |B
    |训练数据集
-|work
    |PaddleOCR
-README.MD
-xxx.ipynb
```
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)
B：此处由项目作者进行撰写使用方式。

## 安装所需的python库
pip install -r requirements.txt

## 开始训练
cd work/PaddleOCR && python3 tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0_2_cutout_suijipinjie_ksize=3_test.yml

## 生成推理模型，测试模型大小
python tools/export_model.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0_2_cutout_suijipinjie_ksize=3.yml -o Global.pretrained_model=./rec_chinese_lite_v2.0_1.0_cutout_suijipinjie_ksize3_test/best_accuracy  Global.save_inference_dir=./inference/rec_inference_lite_cutout_suijipinjie_ksize3_test/

## 对测试集进行推理，并生成可以提交的txt文件
python tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0_2_cutout_suijipinjie_ksize=3.yml -o Global.infer_img="../data/A榜测试数据集/TestAImages" Global.pretrained_model="./output/rec_chinese_lite_v2.0_1.0_cutout_suijipinjie_ksize3_test/best_accuracy"
