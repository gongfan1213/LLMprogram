
import numpy as np
import torch

#device = "cpu"                         #Pytorch的特性，需要指定计算的硬件，如果没有GPU的存在，就使用CPU进行计算
device = "cuda"                         #在这里读者默认使用GPU，如果读者出现运行问题可以将其改成cpu模式

class ToTensor:
    def __call__(self, inputs, targets):   #可调用对象
        inputs = np.reshape(inputs,[28*28])
        return torch.tensor(inputs), torch.tensor(targets)


class MNIST_Datset(torch.utils.data.Dataset):
    def __init__(self,transform = None):    #在定义时需要定义transform的参数
        super(MNIST_Datset, self).__init__()
        # 载入数据
        self.x_train = np.load("../dataset/mnist/x_train.npy")
        self.y_train_label = np.load("../dataset/mnist/y_train_label.npy")

        self.transform = transform          #需要显式的提供transform类

    def __getitem__(self, index):
        image = (self.x_train[index])
        label = (self.y_train_label[index])

        #通过判定transform类的存在对其进行调用
        if self.transform:
            image,label = self.transform(image,label)
        return image,label

    def __len__(self):
        return len(self.y_train_label)

import torch
import numpy as np

batch_size = 320                        #设定每次训练的批次数
epochs = 42                           #设定训练次数

mnist_dataset = MNIST_Datset(transform=ToTensor())
from torch.utils.data import DataLoader
train_loader = DataLoader(mnist_dataset, batch_size=batch_size)

#设定的多层感知机网络模型
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28,312),
            torch.nn.ReLU(),
            torch.nn.Linear(312, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_relu_stack(x)

        return logits

model = NeuralNetwork()
model = model.to(device)                #将计算模型传入GPU硬件等待计算
torch.save(model, './model.pth')
#model = torch.compile(model)            #Pytorch2.0的特性，加速计算速度
loss_fu = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)   #设定优化函数


#开始计算
for epoch in range(epochs):
    train_loss = 0
    for image,label in (train_loader):

        train_image = image.to(device)
        train_label = label.to(device)

        pred = model(train_image)
        loss = loss_fu(pred,train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # 记录每个批次的损失值

    # 计算并打印损失值
    train_loss = train_loss/batch_size
    print("epoch：", epoch, "train_loss:", round(train_loss, 2))


