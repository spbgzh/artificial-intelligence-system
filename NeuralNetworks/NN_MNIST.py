# 导入库
import torch
import torch.nn as nn
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 参数设置
num_input = 28 * 28  # 784
num_hidden = 200
num_class = 10
batch_size = 64
learning_rate = 0.001  # 学习率
num_epochs = 3  # 迭代次数

# 载入数据
Train_datasets = dataset.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
Test_datasets = dataset.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
Train_loader = DataLoader(dataset=Train_datasets, batch_size=batch_size, shuffle=True)
Test_loader = DataLoader(dataset=Test_datasets, batch_size=len(Test_datasets), shuffle=False)

# 设置配置
device = torch.device('mps'if torch.has_mps else 'cpu')


class NN(nn.Module):
    def __init__(self, num_input, num_hidden, num_class):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# 初始化框架
model = NN(num_input, num_hidden, num_class).to(device)

# 自定义损失函数和优化器
LossF = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# 训练
for epoch in range(num_epochs):
    for batch_index, (images, labels) in enumerate(Train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
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
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predict = torch.max(outputs, 1)
        correct_num += (predict == labels).sum()
        total_num += (predict.size(0))
    print("测试集的精度为:{}%".format(correct_num / total_num * 100))

