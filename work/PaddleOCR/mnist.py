import paddle
# 导入并行计算模块
import paddle.distributed as dist
import paddle.vision.transforms as T 

CIFAR100_MEAN = [0.5073715, 0.4867007, 0.441096]
CIFAR100_STD = [0.26750046, 0.25658613, 0.27630225]

train_transfrom = T.Compose([
            T.Resize((118, 118)),
            T.CenterCrop((112, 112)),
            T.RandomHorizontalFlip(0.5),        # 随机水平翻转
            T.RandomRotation(degrees=15),       # （-degrees，+degrees）
            T.ToTensor(),                      # 数据的格式转换和标准化 HWC => CHW  
            T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)  # 图像归一化
        ])

eval_transfrom = T.Compose([
            T.Resize(112),
            T.ToTensor(),                       # 数据的格式转换和标准化 HWC => CHW  
            T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)  # 图像归一化
        ])

train_dataset = paddle.vision.datasets.Cifar100(mode='train', transform=train_transfrom)

# 验证数据集
eval_dataset = paddle.vision.datasets.Cifar100(mode='test', transform=eval_transfrom)
network = paddle.vision.models.resnet101(num_classes=100, pretrained=True)
 
# Mnist继承paddle.nn.Layer属于Net，model包含了训练功能
model=paddle.Model(network)

# 设置训练模型所需的optimizer, loss, metric
model.prepare(
    paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy(topk=(1,2))
)
def train():
    model.fit(train_dataset,epochs=10,batch_size=128,log_freq=40)
    #model.evaluate(test_dataset,log_freq=20,batch_size=64)
if __name__=='__main__':
    dist.spawn(train)