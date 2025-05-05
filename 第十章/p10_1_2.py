import torch
import math
import einops.layers.torch as elt


class Attention(torch.nn.Module):
    def __init__(self,embedding_dim = 312,hidden_dim = 256):
        super().__init__()
        self.query_layer = torch.nn.Linear(embedding_dim, hidden_dim)
        self.key_layer = torch.nn.Linear(embedding_dim, hidden_dim)
        self.value_layer = torch.nn.Linear(embedding_dim, hidden_dim)


    def forward(self,embedding,mask):
        input_embedding = embedding

        query = self.query_layer(input_embedding)
        key = self.key_layer(input_embedding)
        value = self.value_layer(input_embedding)

        key = elt.Rearrange("b l d -> b d l")(key)
        # 计算query与key之间的权重系数
        attention_prob = torch.matmul(query, key)

        # 使用softmax对权重系数进行归一化计算
        attention_prob += mask * -1e5  # 在自注意力权重基础上加上掩模值
        attention_prob = torch.softmax(attention_prob, dim=-1)

        # 计算权重系数与value的值从而获取注意力值
        attention_score = torch.matmul(attention_prob, value)

        return (attention_score)










