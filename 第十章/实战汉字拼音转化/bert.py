# 导入PyTorch库和必要的子模块
import torch
import torch.nn as nn
import math


# 定义Scaled Dot Product Attention类
class Attention(nn.Module):
    """
    计算'Scaled Dot Product Attention'
    """

    # 定义前向传播函数
    def forward(self, query, key, value, mask=None, dropout=None):
        # 通过点积计算query和key的得分，然后除以sqrt(query的维度)进行缩放
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # 如果提供了mask，则对得分应用mask，将mask为0的位置设置为一个非常小的数
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

            # 使用softmax函数计算注意力权重
        p_attn = torch.nn.functional.softmax(scores, dim=-1)

        # 如果提供了dropout，则对注意力权重应用dropout
        if dropout is not None:
            p_attn = dropout(p_attn)

            # 使用注意力权重对value进行加权求和，返回加权后的结果和注意力权重
        return torch.matmul(p_attn, value), p_attn

    # 定义Multi-Head Attention类


class MultiHeadedAttention(nn.Module):
    """
    接受模型大小和注意力头数作为输入。
    """

    # 初始化函数，设置模型参数
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        # 确保d_model可以被h整除
        assert d_model % h == 0

        # 我们假设d_v始终等于d_k
        self.d_k = d_model // h
        self.h = h

        # 创建3个线性层，用于将输入投影到query、key和value空间
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # 创建输出线性层，用于将多头注意力的输出合并到一个向量中
        self.output_linear = nn.Linear(d_model, d_model)
        # 创建注意力机制实例
        self.attention = Attention()

        # 创建dropout层，用于正则化
        self.dropout = nn.Dropout(p=dropout)

        # 定义前向传播函数

    def forward(self, query, key, value, mask=None):
        # 获取batch大小
        batch_size = query.size(0)

        # 对输入进行线性投影，并将结果reshape为(batch_size, h, seq_len, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 对投影后的query、key和value应用注意力机制，得到加权后的结果和注意力权重
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将多头注意力的输出合并到一个向量中，并应用输出线性层
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


# 定义SublayerConnection类，继承自nn.Module
class SublayerConnection(nn.Module):
    """
    该类实现了一个带有层归一化的残差连接。
    为了代码的简洁性，归一化操作被放在了前面，而不是通常的最后。
    """

    def __init__(self, size, dropout):
        # 调用父类的初始化函数
        super(SublayerConnection, self).__init__()
        # 初始化层归一化，size是输入的特征维度
        self.norm = torch.nn.LayerNorm(size)
        # 初始化dropout层，dropout是丢弃率
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        对任何具有相同大小的子层应用残差连接。
        x: 输入张量
        sublayer: 要应用的子层（函数）
        """
        # 首先对x进行层归一化，然后传递给sublayer，再应用dropout，最后与原始x进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))

    # 定义PositionwiseFeedForward类，继承自nn.Module


class PositionwiseFeedForward(nn.Module):
    """
    该类实现了FFN（前馈网络）的公式。这是一个两层的全连接网络。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        # 调用父类的初始化函数
        super(PositionwiseFeedForward, self).__init__()
        # 初始化第一层全连接层，输入维度d_model，输出维度d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 初始化第二层全连接层，输入维度d_ff，输出维度d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # 初始化dropout层，dropout是丢弃率
        self.dropout = nn.Dropout(dropout)
        # 使用GELU作为激活函数
        self.activation = torch.nn.GELU()

    def forward(self, x):
        """
        前向传播函数。输入x经过第一层全连接层、激活函数、dropout层和第二层全连接层。
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    双向编码器 = Transformer (自注意力机制)
    Transformer = 多头注意力 + 前馈网络，并使用子层连接
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: transformer的隐藏层大小
        :param attn_heads: 多头注意力的头数
        :param feed_forward_hidden: 前馈网络的隐藏层大小，通常是4*hidden_size
        :param dropout: dropout率
        """

        super().__init__()  # 调用父类的初始化方法
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)  # 初始化多头注意力模块
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden,
                                                    dropout=dropout)  # 初始化位置相关的前馈网络
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)  # 初始化输入子层连接
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)  # 初始化输出子层连接
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层

    def forward(self, x, mask):  # 前向传播方法
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))  # 对输入x应用注意力机制，并使用输入子层连接
        x = self.output_sublayer(x, self.feed_forward)  # 对x应用前馈网络，并使用输出子层连接
        return self.dropout(x)  # 返回经过dropout处理的x


class PositionalEmbedding(nn.Module):  # 位置嵌入模块，为输入序列提供位置信息

    def __init__(self, d_model, max_len=512):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

        # 在对数空间中一次性计算位置编码
        pe = torch.zeros(max_len, d_model).float()  # 创建一个全0的张量用于存储位置编码
        pe.require_grad = False  # 设置不需要梯度，因为位置编码是固定的，不需要训练

        position = torch.arange(0, max_len).float().unsqueeze(1)  # 创建一个表示位置的张量，从0到max_len-1
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # 计算位置编码的公式中的分母部分

        pe[:, 0::2] = torch.sin(position * div_term)  # 对位置编码的偶数索引应用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 对位置编码的奇数索引应用cos函数

        pe = pe.unsqueeze(0)  # 增加一个维度，以便与输入数据匹配
        self.register_buffer('pe', pe)  # 将位置编码注册为一个buffer，这样它就可以与模型一起移动，但不会被视为模型参数

    def forward(self, x):  # 前向传播方法
        return self.pe[:, :x.size(1)]  # 返回与输入序列长度相匹配的位置编码```


class BERT(nn.Module):
    """
    BERT模型：基于Transformer的双向编码器表示。
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        初始化BERT模型。
        :param vocab_size: 词汇表的大小。
        :param hidden: BERT模型的隐藏层大小，默认为768。
        :param n_layers: Transformer块（层）的数量，默认为12。
        :param attn_heads: 注意力头的数量，默认为12。
        :param dropout: dropout率，默认为0.1。
        """

        super().__init__()  # 调用父类nn.Module的初始化方法。
        self.hidden = hidden  # 保存隐藏层大小。
        self.n_layers = n_layers  # 保存Transformer块的数量。
        self.attn_heads = attn_heads  # 保存注意力头的数量。

        # 论文指出他们使用4*hidden_size作为前馈网络的隐藏层大小。
        self.feed_forward_hidden = hidden * 4  # 计算前馈网络的隐藏层大小。

        # BERT的嵌入，包括位置嵌入、段嵌入和令牌嵌入的总和。
        self.word_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden)  # 创建单词嵌入层。
        self.position_embedding = PositionalEmbedding(d_model=hidden)  # 创建位置嵌入层。

        # 多层Transformer块，深度网络。
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])  # 创建多个Transformer块。

    def forward(self, x):
        """
        前向传播方法。
        :param x: 输入序列，shape为[batch_size, seq_len]。
        :return: 经过BERT模型处理后的输出序列，shape为[batch_size, seq_len, hidden]。
        """

        # 为填充令牌创建注意力掩码。
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len])
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # 创建注意力掩码。
        # 将索引序列嵌入到向量序列中。
        x = self.word_embedding(x) + self.position_embedding(x)  # 将单词嵌入和位置嵌入相加得到输入序列的嵌入表示。

        # 在多个Transformer块上运行。
        for transformer in self.transformer_blocks:  # 遍历所有Transformer块。
            x = transformer.forward(x, mask)  # 将输入序列和注意力掩码传递给每个Transformer块，并获取输出序列。
        return x  # 返回经过所有Transformer块处理后的输出序列。```


if __name__ == '__main__':
    vocab_size = 1024
    seq = arr = torch.tensor([[1,1,1,1,0,0,0],[1,1,1,0,0,0,0]])
    logits = BERT(vocab_size=vocab_size)(seq)
    print(logits.shape)