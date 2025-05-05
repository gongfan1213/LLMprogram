import torch
import math
import einops.layers.torch as elt

class FeedForWard(torch.nn.Module):
    def __init__(self,embdding_dim = 312,scale = 4):
        super().__init__()
        self.linear1 = torch.nn.Linear(embdding_dim,embdding_dim*scale)
        self.relu_1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(embdding_dim*scale,embdding_dim)
        self.relu_2 = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embdding_dim)
    def forward(self,tensor):
        embedding = self.linear1(tensor)
        embedding = self.relu_1(embedding)
        embedding = self.linear2(embedding)
        embedding = self.relu_2(embedding)
        embedding = self.layer_norm(embedding)
        return embedding

class Attention(torch.nn.Module):
    def __init__(self,embedding_dim = 312,hidden_dim = 312,n_head = 6):
        super().__init__()
        self.n_head = n_head
        self.query_layer = torch.nn.Linear(embedding_dim, hidden_dim)
        self.key_layer = torch.nn.Linear(embedding_dim, hidden_dim)
        self.value_layer = torch.nn.Linear(embedding_dim, hidden_dim)


    def forward(self,embedding,mask):
        input_embedding = embedding

        query = self.query_layer(input_embedding)
        key = self.key_layer(input_embedding)
        value = self.value_layer(input_embedding)

        query_splited = self.splite_tensor(query,self.n_head)
        key_splited = self.splite_tensor(key,self.n_head)
        value_splited = self.splite_tensor(value,self.n_head)

        key_splited = elt.Rearrange("b h l d -> b h d l")(key_splited)
        # 计算query与key之间的权重系数
        attention_prob = torch.matmul(query_splited, key_splited)

        # 使用softmax对权重系数进行归一化计算
        attention_prob += mask * -1e5  # 在自注意力权重基础上加上掩模值
        attention_prob = torch.softmax(attention_prob, dim=-1)

        # 计算权重系数与value的值从而获取注意力值
        attention_score = torch.matmul(attention_prob, value_splited)
        attention_score = elt.Rearrange("b h l d -> b l (h d)")(attention_score)

        return (attention_score)

    def splite_tensor(self,tensor,h_head):
        embedding = elt.Rearrange("b l (h d) -> b l h d",h = h_head)(tensor)
        embedding = elt.Rearrange("b l h d -> b h l d", h=h_head)(embedding)
        return embedding


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model = 312, dropout = 0.05, max_len=80):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = torch.nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x) # size = [batch, L, d_model]


class Encoder(torch.nn.Module):
    def __init__(self,vocab_size = 1024,max_length = 80,embedding_size = 312,n_head = 6,scale = 4,n_layer = 3):
        super().__init__()
        self.n_layer = n_layer
        self.embedding_table = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_size)
        self.position_embedding = PositionalEncoding(max_len=max_length)
        self.attention = Attention(embedding_size,embedding_size,n_head)
        self.feedward = FeedForWard()
    def forward(self,token_inputs):
        token = token_inputs
        mask = self.create_mask(token)

        embedding = self.embedding_table(token)
        embedding = self.position_embedding(embedding)
        for _ in range(self.n_layer):
            embedding = self.attention(embedding,mask)
            embedding = torch.nn.Dropout(0.1)(embedding)
            embedding = self.feedward(embedding)

        return embedding

    def create_mask(self,seq):
        mask = torch.not_equal(seq, 0).float()
        mask = torch.unsqueeze(mask, dim=-1)
        mask = torch.unsqueeze(mask, dim=1)
        return mask


if __name__ == '__main__':
    seq = torch.ones(size=(3,80),dtype=int)
    Encoder()(seq)



