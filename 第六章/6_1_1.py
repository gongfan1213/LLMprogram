import numpy as np
import torch

class MNIST_Datset(torch.utils.data.Dataset):
    def __init__(self,transformer = None):  #transformer参数在下一节中会介绍
        super(MNIST_Datset, self).__init__()
        # 载入数据
        self.x_train = np.load("../dataset/mnist/x_train.npy")
        self.y_train_label = np.load("../dataset/mnist/y_train_label.npy")

    def __getitem__(self, item):
        image = (self.x_train[item])
        label = (self.y_train_label[item])
        return image,label
    def __len__(self):
        return len(self.y_train_label)

mnist_dataset = MNIST_Datset()
print(len(mnist_dataset))
image,label = (mnist_dataset[1024])

print(image.shape)
print(label)



