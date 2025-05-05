import torch
import torch.nn as nn
import numpy as np
import einops.layers.torch as elt

class MnistNetword(nn.Module):
    def __init__(self):
        super(MnistNetword, self).__init__()
        self.convs_stack = nn.Sequential(

            nn.Conv2d(1,12,kernel_size=7),  #第一个卷积层
            nn.ReLU(),
            nn.Conv2d(12,24,kernel_size=5),  #第二个卷积层
            nn.ReLU(),
            nn.Conv2d(24,6,kernel_size=3)  #第三个卷积层
        )
        #最终分类器层
        self.logits_layer = nn.Linear(in_features=1536,out_features=10)

    def forward(self,inputs):
        image = inputs
        x = self.convs_stack(image)

        #elt.Rearrange的作用是对输入数据维度进行调整，读者可以使用torch.nn.Flatten函数完成此工作
        x = elt.Rearrange("b c h w -> b (c h w)")(x)
        logits = self.logits_layer(x)
        return logits


model = MnistNetword()
torch.save(model,"model.pth")
