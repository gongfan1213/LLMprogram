import torch

image = torch.randn(size=(5,3,128,128))
#下面是定义的卷积层例子
"""
输入维度：3
输出维度：10
卷积核大小：3
步长：2
补偿方式：维度不变补偿
"""
# conv2d = torch.nn.Conv2d(3,10,kernel_size=3,stride=1,padding=1)
# image_new = conv2d(image)
# print(image_new.shape)


conv2d = torch.nn.Conv2d(3,10,kernel_size=3,stride=2,padding=1)
image_new = conv2d(image)
print(image_new.shape)