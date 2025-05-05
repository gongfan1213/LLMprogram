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
device = "cuda"                         #在这里读者默认使用GPU，如果读者出现运行问题可以将其改成cpu模式

model = unet.Unet()                     #导入Unet模型
model = model.to(device)                #将计算模型传入GPU硬件等待计算
#model = torch.compile(model)            #Pytorch2.0的特性，加速计算速度 选择使用内容
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
train_length = len(x_train) * 20                       #增加数据的单词循环次数

#state_dict = torch.load("./saver/unet.pth")
#model.load_state_dict(state_dict)
for epoch in range(30):
    train_num = train_length // batch_size             #计算有多少批次数

    train_loss = 0                                     #用于损失函数的统计
    for i in tqdm(range(train_num)):                    #开始循环训练
        x_imgs_batch = []                               #创建数据的临时存储位置
        x_step_batch = []
        y_batch = []
        # 对每个批次内的数据进行处理
        for b in range(batch_size):
            img = x_train[np.random.randint(x_train.shape[0])]  #提取单个图片内容
            x = img
            y = img

            x_imgs_batch.append(x)
            y_batch.append(y)

        #将批次数据转化为Pytorch对应的tensor格式并将其传入GPU中
        x_imgs_batch = torch.tensor(x_imgs_batch).float().to(device)
        y_batch = torch.tensor(y_batch).float().to(device)


        pred = model(x_imgs_batch)                      #对模型进行正向计算
        loss = torch.nn.MSELoss(reduction="sum")(pred, y_batch)*100.   #使用损失函数进行计算

        #这里读者记住下面就是固定格式，一般而言这样使用即可
        optimizer.zero_grad()                                               #对结果进行优化计算
        loss.backward()                                                     #损失值的反向传播
        optimizer.step()                                                    #对参数进行更新

        train_loss += loss.item()                                           #记录每个批次的损失值
    #计算并打印损失值
    train_loss /= train_num
    print("train_loss:", train_loss)
    if epoch%6 == 0:
        torch.save(model.state_dict(),"./saver/unet.pth")

    #下面是对数据进行打印
    image = x_train[np.random.randint(x_train.shape[0])]                    #随机挑选一条数据进行计算
    image = np.reshape(image,[1,1,28,28])                                   #修正数据维度

    image = torch.tensor(image).float().to(device)                          #挑选的数据传入硬件中等待计算
    image = model(image)                                                    #使用模型对数据进行计算

    image = torch.reshape(image, shape=[28,28])                             #修正模型输出结果
    image = image.detach().cpu().numpy()                                    #将计算结果导入CPU中进行后续计算或者展示

    #展示或计算数据结果
    plt.imshow(image)
    plt.savefig(f"./img/img_{epoch}.jpg")


