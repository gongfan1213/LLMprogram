import torch
import math
import einops.layers.torch as elt
# word_embedding_table = torch.nn.Embedding(num_embeddings=encoder_vocab_size,embedding_dim=312)
# encoder_embedding = word_embedding_table(inputs)


vocab_size = 1024   #字符的种类
embedding_dim = 312
hidden_dim = 256
token = torch.ones(size=(5,80),dtype=int)
#创建一个输入embedding值
input_embedding = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)(token)

#对输入的input_embedding进行修正，这里进行了简写
query = torch.nn.Linear(embedding_dim,hidden_dim)(input_embedding)
key = torch.nn.Linear(embedding_dim,hidden_dim)(input_embedding)
value = torch.nn.Linear(embedding_dim,hidden_dim)(input_embedding)

key = elt.Rearrange("b l d -> b d l")(key)
#计算query与key之间的权重系数
attention_prob = torch.matmul(query,key)

#使用softmax对权重系数进行归一化计算
attention_prob = torch.softmax(attention_prob,dim=-1)

#计算权重系数与value的值从而获取注意力值
attention_score = torch.matmul(attention_prob,value)

print(attention_score.shape)