"""
    这是输出文件，需要首先运行train文件
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #指定GPU编
import torch
import numpy as np
import unet
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size = 320                        #设定每次训练的批次数
epochs = 1024                           #设定训练次数

#device = "cpu"                         #Pytorch的特性，需要指定计算的硬件，如果没有GPU的存在，就使用CPU进行计算
device = "cpu"                         #在这里读者默认使用GPU，如果读者出现运行问题可以将其改成cpu模式

model = unet.Unet()                     #导入Unet模型
model = model.to(device)                #将计算模型传入GPU硬件等待计算
#model = torch.compile(model)            #Pytorch2.0的特性，加速计算速度 在部分机器上无法运行
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)   #设定优化函数

#载入数据
x_train = np.load("../dataset/mnist/x_train.npy")
y_train_label = np.load("../dataset/mnist/y_train_label.npy")

x_train_batch = []
for i in range(len(y_train_label)):
    if y_train_label[i] <= 10:                    #为了加速演示作者只对数据集中的小于2的数字，也就是0和1进行运行，读者可以自行增加训练个数
        x_train_batch.append(x_train[i])

x_train = np.reshape(x_train_batch, [-1, 1, 28, 28])  #修正数据输入维度：([30596, 28, 28])
x_train /= 512.

image = x_train[np.random.randint(28)]                    #随机挑选一条数据进行计算
image = np.reshape(image,[28,28])                                   #修正数据维度
plt.imshow(image)
plt.show()

state_dict = torch.load("./saver/unet.pth")
model.load_state_dict(state_dict)
image = torch.reshape(torch.tensor(image),[1,1,28,28])
img = model(image)
img = torch.reshape(img, shape=[28,28])                             #修正模型输出结果
img = img.detach().cpu().numpy()
plt.imshow(img)
plt.show()



