
import torch
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

#创建模型
model = NeuralNetwork()

# 模拟输入数据
input_data = (torch.rand(5, 784))

from tensorboardX import SummaryWriter
writer = SummaryWriter()

with writer:
    writer.add_graph(model,(input_data,))

