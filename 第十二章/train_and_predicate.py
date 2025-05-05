# 导入库
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import einops.layers.torch as elt

import get_dataset_v2
from tqdm import tqdm

sentences = get_dataset_v2.sentences
src_vocab = get_dataset_v2.src_vocab
tgt_vocab = get_dataset_v2.tgt_vocab

src_vocab_size = len(src_vocab) #4462
tgt_vocab_size  = len(tgt_vocab)    #1154

src_len = 48
tgt_len = 47    #由于输出的比输入多一个符号，就用这个
# ***********************************************#
# transformer的参数
# Transformer Parameters
d_model = 312
# 每一个词的 word embedding 用多少位表示
# （包括positional encoding应该用多少位表示，因为这两个要维度相加，应该是一样的维度）
d_ff = 2048  # FeedForward dimension
# forward线性层变成多少位(d_model->d_ff->d_model)
d_k = d_v = 64  # dimension of K(=Q), V
# K，Q，V矩阵的维度（K和Q一定是一样的，因为要K乘Q的转置），V不一定
'''
换一种说法，就是我在进行self-attention的时候，
从input（当然是加了位置编码之后的input）线性变换之后的三个向量 K，Q，V的维度
'''
n_layers = 6
# encoder和decoder各有多少层
n_heads = 8


# multi-head attention有几个头
# ***********************************************#

# 数据预处理
#	将encoder_input、decoder_input和decoder_output进行id化

enc_inputs, dec_inputs, dec_outputs = [], [], []
for line in sentences:
    enc = line[0];dec_in = line[1];dec_tgt = line[2]
    if len(enc) <= src_len and len(dec_in) <= tgt_len and len(dec_tgt) <= tgt_len:


        enc_token = [src_vocab[char] for char in enc];enc_token = enc_token + [0] * (src_len - len(enc_token))
        dec_in_token = [tgt_vocab[char] for char in dec_in];dec_in_token = dec_in_token + [0] * (tgt_len - len(dec_in_token))
        dec_tgt_token = [tgt_vocab[char] for char in dec_tgt];dec_tgt_token = dec_tgt_token + [0] * (tgt_len - len(dec_tgt_token))

        enc_inputs.append(enc_token);dec_inputs.append(dec_in_token);dec_outputs.append(dec_tgt_token)

enc_inputs = torch.LongTensor(enc_inputs)
dec_inputs = torch.LongTensor(dec_inputs)
dec_outputs = torch.LongTensor(dec_outputs)
# print(enc_inputs[0])
# print(dec_inputs[0])
# print(dec_outputs[0])


# ***********************************************#
print(enc_inputs.shape,dec_inputs.shape,dec_outputs.shape)

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    def __len__(self):
        return self.enc_inputs.shape[0]
    # 有几个sentence
    def __getitem__(self, idx):


        return torch.tensor(self.enc_inputs[idx]), torch.tensor(self.dec_inputs[idx]), torch.tensor(self.dec_outputs[idx])
    # 根据索引找encoder_input,decoder_input,decoder_output


loader = Data.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs),
    batch_size=4,
    shuffle=True)


# ***********************************************#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # max_length_（一个sequence的最大长度）
        pe = torch.zeros(max_len, d_model)
        # pe [max_len,d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # position  [max_len，1]

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        # div_term:[d_model/2]
        # e^(-i*log10000/d_model)=10000^(-i/d_model)
        # d_model为embedding_dimension

        # 两个相乘的维度为[max_len,d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 计算position encoding
        # pe的维度为[max_len,d_model],每一行的奇数偶数分别取sin和cos(position * div_term)里面的值
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 维度变成(max_len,1,d_model)
        # 所以直接用pe=pe.unsqueeze(1)也可以
        self.register_buffer('pe', pe)
        # 放入buffer中，参数不会训练

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :, :]
        # 选取和x一样维度的seq_length，将pe加到x上
        return self.dropout(x)
    # ***********************************************#


# 由于在 Encoder 和 Decoder 中都需要进行 mask 操作，
# 因此就无法确定这个函数的参数中 seq_len 的值，
# 如果是在 Encoder 中调用的，seq_len 就等于 src_len
# 如果是在 Decoder 中调用的，seq_len 就有可能等于 src_len，
# 也有可能等于 tgt_len（因为 Decoder 有两次 mask）
# src_len 是在encoder-decoder中的mask
# tgt_len是decdoer mask

def creat_self_mask(from_tensor, to_tensor):
    """
      这里需要注意，from_tensor 是输入的文本序列，即 input_word_ids ，应该是2D的，即[1,2,3,4,5,6,0,0,0,0]
                  to_tensor 是输入的的 input_word_ids，应该是2D的，即[1,2,3,4,5,6,0,0,0,0]

                  最终的结果是输出2个3D的相乘，
                注意：后面如果需要4D的，则使用expand添加一个维度即可
    """
    batch_size, from_seq_length = from_tensor.shape
    # 这里只能做self attention，不能做交互
    # assert from_tensor == to_tensor,print("输入from_tensor与to_tensor不一致，检查mask创建部分,需要自己完成")

    # 注意这里的数据类型转换方法，from：https://wenku.baidu.com/view/e2e67e2eb868a98271fe910ef12d2af90242a86e.html?_wkts_=1672014945065&bdQuery=torch%E8%BD%AC%E5%8C%96%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B
    to_mask = torch.not_equal(from_tensor, 0).int()
    to_mask = elt.Rearrange("b l -> b 1 l")(to_mask)  # 这里扩充了数据类型

    broadcast_ones = torch.ones_like(to_tensor)
    broadcast_ones = torch.unsqueeze(broadcast_ones, dim=-1)

    # print(broadcast_ones.is_cuda)
    # print(to_mask.is_cuda)
    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask
    mask.to("cuda")
    return mask

def create_look_ahead_mask(from_tensor, to_tensor):
    corss_mask = creat_self_mask(from_tensor, to_tensor)
    look_ahead_mask = torch.tril(torch.ones(to_tensor.shape[1], from_tensor.shape[1]))
    look_ahead_mask = look_ahead_mask.to("cuda")


    corss_mask = look_ahead_mask * corss_mask
    return corss_mask

# ***********************************************#
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask == 0, -1e9)
        # attn_mask所有为True的部分（即有pad的部分），scores填充为负无穷，也就是这个位置的值对于softmax没有影响
        attn = nn.Softmax(dim=-1)(scores)
        # attn： [batch_size, n_heads, len_q, len_k]
        # 对每一行进行softmax
        context = torch.matmul(attn, V)
        # [batch_size, n_heads, len_q, d_v]
        return context, attn


'''
这里要做的是，通过 Q 和 K 计算出 scores，然后将 scores 和 V 相乘，得到每个单词的 context vector
第一步是将 Q 和 K 的转置相乘没什么好说的，相乘之后得到的 scores 还不能立刻进行 softmax，
需要和 attn_mask 相加，把一些需要屏蔽的信息屏蔽掉，
attn_mask 是一个仅由 True 和 False 组成的 tensor，并且一定会保证 attn_mask 和 scores 的维度四个值相同（不然无法做对应位置相加）
mask 完了之后，就可以对 scores 进行 softmax 了。然后再与 V 相乘，得到 context
'''
# ***********************************************#
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 三个矩阵，分别对输入进行三次线性变化
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        # 变换维度

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        #  [batch_size, len_q, d_model]
        # (W)-> [batch_size, len_q,d_k * n_heads]
        # (view)->[batch_size, len_q,n_heads,d_k]
        # (transpose)-> [batch_size,n_heads, len_q,d_k ]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # 生成Q，K，V矩阵

        attn_mask = attn_mask.unsqueeze(1)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # context: [batch_size, n_heads, len_q, d_v],
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)
        # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn

'''
完整代码中一定会有三处地方调用 MultiHeadAttention()，Encoder Layer 调用一次，
传入的 input_Q、input_K、input_V 全部都是 enc_inputs；
Decoder Layer 中两次调用，第一次都是decoder_inputs；第二次是两个encoder_outputs和一个decoder——input
'''
# ***********************************************#
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
    # 也有残差连接和layer normalization
    # 这段代码非常简单，就是做两次线性变换，残差连接后再跟一个 Layer Norm

# ***********************************************#
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()
        # 提取特征

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''

        # enc_outputs: [batch_size, src_len, d_model],
        # attn: [batch_size, n_heads, src_len, src_len] 每一个投一个注意力矩阵
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_inputs to same Q,K,V
        # 乘以WQ，WK，WV生成QKV矩阵（为什么传三个？因为这里传的是一样的
        # 但在decoder-encoder的mulit-head里面，我们需要的decoder input encoder output encoder output
        # 所以为了使用方便，我们在定义enc_self_atten函数的时候就定义的使有三个形参的

        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        # 输入和输出的维度是一样的
        return enc_outputs, attn

# 将上述组件拼起来，就是一个完整的 Encoder Layer
# ***********************************************#
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        # 先是decoder的self-attention

        # print(dec_outputs.shape)
        # print(enc_outputs.shape)
        #
        # print(dec_enc_attn_mask.shape)

        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # 再是encoder-decoder attention部分

        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        # 特征提取
        return dec_outputs, dec_self_attn, dec_enc_attn

# 在 Decoder Layer 中会调用两次 MultiHeadAttention，第一次是计算 Decoder Input 的 self-attention，得到输出 dec_outputs。
# 然后将 dec_outputs 作为生成 Q 的元素，enc_outputs 作为生成 K 和 V 的元素，再调用一次 MultiHeadAttention，得到的是 Encoder 和 Decoder Layer 之间的 context vector。最后将 dec_outptus 做一次维度变换，然后返回
# ***********************************************#
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 对encoder的输入的每个单词进行词向量计算词向量/字向量（src——vocab_size个词，每个词d_model的维度)

        self.pos_emb = PositionalEncoding(d_model)
        # 计算位置向量

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        # 将6个EncoderLayer组成一个module

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)
        # 对每个单词进行词向量计算
        # enc_outputs [batch_size, src_len, d_model]

        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # 添加位置编码
        #  enc_outputs [batch_size, src_len, d_model]

        enc_self_attn_mask = creat_self_mask(enc_inputs, enc_inputs)
        # enc_self_attn: [batch_size, src_len, src_len]
        # 计算得到encoder-attention的pad martix

        enc_self_attns = []
        # 创建一个列表，保存接下来要返回的字-字attention的值，不参与任何计算，供可视化用

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
            # 再传进来就不用positional decoding
            # 记录下每一次的attention
        return enc_outputs, enc_self_attns

# 使用 nn.ModuleList() 里面的参数是列表，列表里面存了 n_layers 个 Encoder Layer

# 由于我们控制好了 Encoder Layer 的输入和输出维度相同，所以可以直接用个 for 循环以嵌套的方式，
# 将上一次 Encoder Layer 的输出作为下一次 Encoder Layer 的输入
# ***********************************************#
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model] 经过六次encoder之后得到的东西
        '''
        dec_outputs = self.tgt_emb(dec_inputs)
        # [batch_size, tgt_len, d_model]
        # 同样地，对decoder_layer进行词向量的生成
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()
        # 计算他的位置向量
        # [batch_size, tgt_len, d_model]

        dec_self_attn_mask = creat_self_mask(dec_inputs, dec_inputs)
        # [batch_size, tgt_len, tgt_len]

        #dec_self_attn_subsequence_mask = create_look_ahead_mask(dec_inputs).cuda()
        # [batch_size, tgt_len, tgt_len]
        # 当前时刻我是看不到未来时刻的东西的

        dec_enc_attn_mask = create_look_ahead_mask(enc_inputs,dec_inputs)
        # [batch_size, tgt_len, tgt_len]
        # 布尔+int  false 0 true 1，gt 大于 True
        # 这样把dec_self_attn_pad_mask和dec_self_attn_subsequence_mask里面为True的部分都剔除掉了
        # 也就是说，即屏蔽掉了pad也屏蔽掉了mask

        # 在decoder的第二个attention里面使用
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# ***********************************************#

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        #self.encoder = Encoder().cuda()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
        # 对decoder的输出转换维度，
        # 从隐藏层维数->英语单词词典大小（选取概率最大的那一个，作为我们的预测结果）

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs维度：[batch_size, src_len]
        对encoder-input，我一个batch中有几个sequence，一个sequence有几个字
        dec_inputs: [batch_size, tgt_len]
        对decoder-input，我一个batch中有几个sequence，一个sequence有几个字
        '''
        # enc_outputs: [batch_size, src_len, d_model]，
        # d_model是每一个字的word embedding长度
        """
        enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        注意力矩阵，对encoder和decoder，每一层，每一句话，每一个头，每两个字之间都有一个权重系数，
        这些权重系数组成了注意力矩阵(之后的dec_self_attns同理，当然decoder还有一个decoder-encoder的矩阵)
        """
        enc_outputs,_ = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.projection(dec_outputs)
        # 将输出的维度，从 [batch_size, tgt_len, d_model]变成[batch_size, tgt_len, tgt_vocab_size]
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1))


# dec_logits 的维度是 [batch_size * tgt_len, tgt_vocab_size]，可以理解为，
# 一个句子，这个句子有 batch_size*tgt_len 个单词，每个单词有 tgt_vocab_size 种情况，取概率最大者
# Transformer 主要就是调用 Encoder 和 Decoder。最后返回
# ***********************************************#
save_path = "./saver/transformer.pt"
device = "cuda"
model = Transformer()
model.to(device)
#model.load_state_dict(torch.load(save_path))
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
# ***********************************************#
for epoch in range(60):
    pbar = tqdm((loader), total=len(loader))  # 显示进度条
    for enc_inputs, dec_inputs, dec_outputs in pbar:

        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]

        outputs = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"epoch {epoch + 1} : train loss {loss.item():.6f} ")  # : learn_rate {lr_scheduler.get_last_lr()[0]:.6f}

    torch.save(model.state_dict(), save_path)

idx2pinyin = {i: w for i, w in enumerate(tgt_vocab)}
idx2hanzi = {i: w for i, w in enumerate(src_vocab)}

context = "从前有座山山里有座庙"
token = [src_vocab[char] for char in context]
token = torch.tensor(token)
sentence_tensor = torch.LongTensor(token).unsqueeze(0).to(device)
outputs = [1]
for i in range(tgt_len):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)

        with torch.no_grad():
            output= model(sentence_tensor, trg_tensor)
        best_guess  = torch.argmax(output,dim=-1).detach().cpu()

        outputs.append(best_guess[-1])
        # if best_guess[-1] == 2:
        #     break
print([idx2pinyin[id.item()] for id in outputs[1:]])




