import torch

image = torch.randn(size=(5,3,28,28))

pool = torch.nn.AvgPool2d(kernel_size=3,stride=2,padding=0)
image_pooled = pool(image)
print(image_pooled.shape)

image_pooled = torch.nn.AdaptiveAvgPool2d(1)(image)
print(image_pooled.shape)