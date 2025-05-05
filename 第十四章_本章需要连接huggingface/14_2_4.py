import torch
from _14_1_2 import *

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        # 此时nx=n_embed=768；
        # 而n_state实际为inner_dim，即n_state为4 * n_embd等于3072。
        nx = config.n_embd

        # self.c_fc = Conv1D(n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
        # n_state(3072), 此时n_state=3072.
        self.c_fc = Conv1D(n_state, nx)
        # self.c_proj = Conv1D(nx, n_state)相当于全连接层, 其将输入张量的最后一个维度的维度数由n_state(3072)投影为
        # nx(768), 此时n_state=3072.
        self.c_proj = Conv1D(nx, n_state)

        # 激活函数gelu.
        self.act = torch.nn.GELU()
        # 残差dropout层进行正则化操作, 防止过拟合.
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
