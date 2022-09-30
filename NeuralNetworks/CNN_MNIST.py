# 导入库
import torch
import torch.nn as nn
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 参数设置
in_channel = 1
features = 8
num_class = 10
batch_size = 64
learning_rate = 0.001  # 学习率
num_epochs = 5  # 迭代次数

# 载入数据
Train_datasets = dataset.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
Test_datasets = dataset.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
Train_loader = DataLoader(dataset=Train_datasets, batch_size=batch_size, shuffle=True)
Test_loader = DataLoader(dataset=Test_datasets, batch_size=len(Test_datasets), shuffle=False)

# 设置配置
device = torch.device('mps'if torch.has_mps else 'cpu')


class CNN(nn.Module):
    def __init__(self, in_channel, features, num_class):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(in_channel, features, kernel_size=3, stride=1, padding=1)  # B*1*28*28 --> B*8*28*28
        self.Relu1 = nn.ReLU()
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # B*8*28*28 --> B*8*14*14
        self.Conv2 = nn.Conv2d(features, features * 2, kernel_size=3, stride=1, padding=1)  # B*8*14*14 --> B*16*14*14
        self.Relu2 = nn.ReLU()
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # B*16*14*14 --> B*16*7*7
        self.Fc = nn.Linear(16 * 7 * 7, num_class)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu1(x)
        x = self.Pool1(x)
        x = self.Conv2(x)
        x = self.Relu2(x)
        x = self.Pool2(x)
        x = x.reshape(x.shape[0], -1)  # [B, 16, 7, 7] --> [B, 16*7*7]
        x = self.Fc(x)
        return x


# 初始化框架
model = CNN(in_channel, features, num_class).to(device)

# 自定义损失函数和优化器
LossF = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# 训练
for epoch in range(num_epochs):
    for batch_index, (images, labels) in enumerate(Train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # 计算损失
        loss = LossF(outputs, labels)

        # 梯度向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 0:
            print('{}/{},[{}/{}],loss={:.4f}'.format(epoch, num_epochs, batch_index, len(Train_loader), loss))
# 测试
with torch.no_grad():
    correct_num = 0
    total_num = 0
    for images, labels in Test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predict = torch.max(outputs, 1)
        correct_num += (predict == labels).sum()
        total_num += (predict.size(0))
    print("测试集的精度为:{}%".format(correct_num / total_num * 100))
