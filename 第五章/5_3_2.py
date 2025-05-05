import torch
from torch.nn import Conv2d,Linear

linear = Linear(in_features=3*28*28, out_features=3*28*28)
linear_params = sum(p.numel() for p in linear.parameters() if p.requires_grad)

conv = Conv2d(in_channels=3, out_channels=3, kernel_size=3)
params = sum(p.numel() for p in conv.parameters() if p.requires_grad)

depth_conv = Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3)
point_conv = Conv2d(in_channels=3, out_channels=3, kernel_size=1)

# 需要注意的是：这里是先搞depth，然后搞逐点卷积，从而两者结合，就得到 了
# 深度，分离，卷积
depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)

print(f"多层感知机使用参数为 {params} parameters.")
print("----------------")
print(f"普通卷积层使用参数为 {params} parameters.")
print("----------------")
print(f"深度可分离卷积使用参数为 {params_depthwise} parameters.")
