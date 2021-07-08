# AI-Studio-项目标题

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
pip install -r requirements.txt

## 生成推理模型，测试模型大小
pip install -r requirements.txt

## 对测试集进行推理，并生成可以提交的txt文件
pip install -r requirements.txt
