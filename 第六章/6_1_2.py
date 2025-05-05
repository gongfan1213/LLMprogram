import numpy as np

import torch


class ToTensor:
    def __call__(self, inputs, targets):   #可调用对象
        inputs = np.reshape(inputs,[1,-1])
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

mnist_dataset = MNIST_Datset()
image,label = (mnist_dataset[1024])
print(type(image), type(label))
print("----------------------------------")
mnist_dataset = MNIST_Datset(transform=ToTensor())
image,label = (mnist_dataset[1024])
print(type(image), type(label))
print(image.shape)



