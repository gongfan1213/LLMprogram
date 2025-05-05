import torch
from _14_1_2 import *

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,prune_conv1d_layer
)

class Attention(torch.nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        # 利用断言函数判断此时隐藏状态的维度数n_state除以注意力头数config.n_head之后是否能整除.
        assert n_state % config.n_head == 0

        # 下方的self.register_buffer()函数的操作相当于创建了两个Attention类中的self属性, 即为self.bias属性
        # 与self.masked_bias属性；
        # 其中self.bias属性为一个下三角矩阵(对角线下元素全为1, 对角线上元素全为0), 其形状为(1, 1, n_ctx, n_ctx),
        # 也即形状相当于(1, 1, 1024, 1024)；
        # 而self.masked_bias属性则为一个极大的负数-1e4；
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            # self.c_attn = Conv1D(2 * n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
            # 2 * n_state(2*768), 此时n_state = nx = num_head*head_features = 768.
            self.c_attn = Conv1D(2 * n_state, nx)

            # self.q_attn = Conv1D(n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
            # n_state(768), 此时n_state = nx = num_head*head_features = 768.
            self.q_attn = Conv1D(n_state, nx)

        else:
            # self.c_attn = Conv1D(3 * n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
            # 2 * n_state(2*768), 此时n_state = nx = num_head*head_features = 768.
            self.c_attn = Conv1D(3 * n_state, nx)

        # 此处self.c_proj()为Conv1D(n_state, nx)函数(all_head_size=n_state=nx=768), 相当于一个全连接层的作用,
        # 其将此时的多头注意力聚合操作结果张量a的最后一个维度all_head_size由n_state(768)的维度数投影为nx(768)的维度数.
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    # prune_heads()可结合 https://github.com/huggingface/transformers/issues/850 理解.
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def merge_heads(self, x):
        # 此时x为: 利用计算得到的注意力分数张量对value张量进行注意力聚合后得到的注意力结果张量.
        # x的形状为(batch_size, num_head, sql_len, head_features).

        # 此时先将注意力结果张量x的形状变为(batch_size, sql_len, num_head, head_features)
        x = x.permute(0, 2, 1, 3).contiguous()
        # new_x_shape为(batch_size, sql_len, num_head*head_features) =》(batch_size, sql_len, all_head_size)
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)

        # 此时将注意力结果张量x的注意力头维度num_head与注意力特征维度head_features进行合并变为all_head_size维度,
        # 注意力结果张量x的形状变为(batch_size, sql_len, all_head_size).
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states， (batch_size, sql_len, all_head_size).

    def split_heads(self, x, k=False):
        # 此时new_x_shape为: (batch_size, sql_len, num_head, head_features)
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # 将输入的张量x(可能为query、key、value张量)变形为: (batch_size, sql_len, num_head, head_features).
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states

        # 若此时输入的张量为key张量,则需要将key张量再变形为(batch_size, num_head, head_features, sql_len).
        # 因为此时key张量需要以[query * key]的形式与query张量做内积运算, 因此key张量需要将head_features变换到第三维度,
        # 将sql_len变换到第四维度,这样[query * key]内积运算之后的注意力分数张量的形状才能符合(batch_size, num_head, sql_len, sql_len).
        if k:
            return x.permute(0, 2, 3, 1)  # (batch_size, num_head, head_features, sql_len)

        # 若此时输入的张量为query张量或value张量, 则将张量维度再变换为(batch_size, num_head, sql_len, head_features)即可,
        # 即将sql_len与num_head调换维度.
        else:
            return x.permute(0, 2, 1, 3)  # (batch_size, num_head, sql_len, head_features)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):

        '''
        此时query张量形状为: (batch_size, num_head, 1, head_features)
        key张量的形状为: (batch_size, num_head, head_features, sql_len+1)
        value张量的形状为: (batch_size, num_head, sql_len+1, head_features)

        此时key张量以[query * key]的形式与query张量做内积运算, key张量已在split_heads()操作与past_key合并操作中
        提前将head_features变换到第三维度, 将sql_len+1变换到第四维度,这样[query * key]内积运算之后的注意力分数张量w的
        形状才能符合(batch_size, num_head, 1, sql_len+1).
        '''
        w = torch.matmul(q, k)  # 注意力分数张量w: (batch_size, num_head, 1, sql_len+1)

        # 对注意力分数张量w中的值进行缩放(scaled), 缩放的除数为注意力头特征数head_features的开方值.
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        # 此时nd与ns两个维度相当于1与seq_len+1
        nd, ns = w.size(-2), w.size(-1)

        # 此处的操作为利用torch.where(condition, x, y)函数,将注意力分数张量w在mask.bool()条件张量为True(1)的相同位置的值
        # 保留为w中的原值, 将在mask.bool()条件张量为True(0)的相同位置的值变为self.masked_bias(-1e4)的值.
        '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时nd与ns维度才会相等, 
        在nd与ns维度相等的情况下此操作的结果等价于让注意力分数张量w与attention_mask张量相加的结果。
        <2> 若为GPT2Mode第二次及之后的迭代时, nd与ns两个维度相当于1与seq_len+1, 此时对self.bias进行切片操作时, 
        ns - nd等于seq_len+1 - 1即结果为seq_len, 即此时切片操作相当于self.bias[:, :, seq_len : seq_len+1, :seq_len+1],
        此操作的意义在于对此次迭代中, 最新的token的注意力分数上添加GPT2中的下三角形式的注意力遮罩.'''
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # 此时self.bias属性为一个下三角矩阵(对角线下元素全为1, 对角线上元素全为0), 其形状为(1, 1, n_ctx, n_ctx),
            # 也即形状相当于(1, 1, 1024, 1024)；但此处对self.bias进行切片操作时, ns - nd等于seq_len+1 - 1即结果为seq_len,
            # 即此时切片操作相当于self.bias[:, :, seq_len : seq_len+1, :seq_len+1]。
            '''此时mask张量(经过大张量self.bias切片获得)的形状为(1, 1, 1, seq_len + 1).'''
            mask = self.bias[:, :, ns - nd: ns, :ns]
            '''此操作的意义在于对此次迭代中, 最新的token的注意力分数上添加GPT2中的下三角形式注意力遮罩.'''
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        # 让注意力分数张量w与attention_mask张量相加, 以达到让填充特殊符[PAD]处的注意力分数为一个很大的负值的目的,这样在下面将
        # 注意力分数张量w输入Softmax()层计算之后, 填充特殊符[PAD]处的注意力分数将会变为无限接近0的数, 以此让填充特殊符[PAD]
        # 处的注意力分数极小, 其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        # 注意力分数张量w: (batch_size, num_head, 1, sql_len+1).
        # 将注意力分数张量w输入进Softmax()层中进行归一化计算, 计算得出最终的注意力分数,
        # 再将注意力分数张量w输入进Dropout层self.attn_dropout()中进行正则化操作, 防止过拟合.
        w = torch.nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to, 对注意力头num_head维度的mask操作.
        if head_mask is not None:
            w = w * head_mask

        # 多头注意力聚合操作: 注意力分数张量w与value张量进行内积
        # 注意力分数张量w形状: (batch_size, num_head, 1, sql_len+1)
        # value张量形状: (batch_size, num_head, sql_len+1, head_features)
        # 多头注意力聚合操作结果张量形状: (batch_size, num_head, 1, head_features), head_features=768.
        outputs = [torch.matmul(w, v)]
        # 若同时返回注意力分数张量w, 则将w张量添加入outputs列表中.
        if output_attentions:
            outputs.append(w)

        return outputs

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        # <1> 此时的隐藏状态hidden_states的形状为 (batch_size, 1, nx), 此时nx = n_state = n_embed = head_features = 768，
        #     即此时隐藏状态hidden_states的形状为(batch_size, 1, 768)。
        # <2> 此时layer_past为一个存储着past_key张量与past_value张量的大张量, 其
        #     形状为(2, batch_size, num_head, sql_len, head_features).
        # <3> attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,
        #     其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.

        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. " \
               "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."

            '''self.crossattention()的Cross_Attention运算过程则是将LayerNormalization之后的hidden_states通过
            'self.q_attn = Conv1D(n_state, nx)(第168行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 768),
            将此投影之后的hidden_states赋值作为query张量；
            再将此时从编码器(encoder)中传过来的编码器隐藏状态encoder_hidden_states通过'self.c_attn = Conv1D(2 * n_state, nx)
            (第164行代码)'将encoder_hidden_states的形状由(batch_size, enc_seq_len, 768)投影为(batch_size, enc_seq_len, 2 * 768),
            将投影后的encoder_hidden_states在在第三维度(dim=2)上拆分为两份分别赋为key、value,
            其形状都为(batch_size, enc_seq_len, 768)；  此时n_state = nx = num_head*head_features = 768.

            之后经过split_heads()函数拆分注意力头之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, enc_seq_len),
            value张量的形状变为(batch_size, num_head, enc_seq_len, head_features).

            此时计算出的cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).'''

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask

        else:
            '''此时隐藏状态hidden_states的形状为(batch_size, 1, 768), 将其输入进全连接层self.c_attn中后,
            其Conv1D(3 * n_state, nx)操作(nx=n_state=768)便会将hidden_states的第三维度数由 768维 投影为 3 * 768维,
            此时的hidden_states张量的形状为(batch_size, 1, 3 * 768), 最后将hidden_states张量在第三个维度(维度数3 * 768)上
            切分为三块, 将这切分出的三块各当成query, key, value张量, 则每个张量的形状都为(batch_size, 1, 768).
            此时n_state = nx = num_head*head_features = 768.

            之后经过split_heads()函数拆分注意力头且key、value张量分别与past_key、past_value张量合并之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, sql_len+1),
            value张量的形状变为(batch_size, num_head, sql_len+1, head_features).'''
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
        # 此时经过'注意力头拆分函数split_heads()'之后的query、key、value三个张量的形状分别为:
        # query: (batch_size, num_head, 1, head_features)
        # key: (batch_size, num_head, head_features, 1)
        # value: (batch_size, num_head, 1, head_features)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
            '''<1> 本次迭代中新的key张量
            此时需要通过layer_past[0].transpose(-2, -1)操作将past_key张量的形状变为(batch_size, num_head, head_features, sql_len),
            而此时key张量的形状为(batch_size, num_head, head_features, 1), 这样在下方就方便将past_key张量与key张量在最后
            一个维度(dim=-1)处进行合并, 这样就将当前token的key部分加入了past_key的seq_len中, 以方便模型在后面预测新的token,
            此时新的key张量的形状为: (batch_size, num_head, head_features, sql_len+1), new_seq_len为sql_len+1。
             <2> 本次迭代中新的value张量
            而此时past_value不用变形, 其形状为(batch_size, num_head, sql_len, head_features), 而此时value张量的形状为
            (batch_size, num_head, 1, head_features), 这样在下方就方便将past_value张量与value张量在倒数第二个
            维度(dim=-2)处进行合并, 这样就将当前token的value部分加入了past_value的seq_len中, 以方便模型在后面预测新的token,
            此时新的value张量的形状为: (batch_size, num_head, sql_len+1, head_features), new_seq_len为sql_len+1。
           '''
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        # config对应的GPT2Config()类中的use_cache默认为True.但此时若为Cross_Attention运算过程, 则此时不会指定use_cache,
        # 而此时use_cache属性即为False(因为Attention类中use_cache属性默认为False, 除非指定config对应的GPT2Config()类
        # 中的use_cache属性其才会为True).
        if use_cache is True:
            # 若use_cache为True, 此时将key张量的最后一个维度与倒数第二个维度互换再与value张量进行stack合并,
            # 此时key.transpose(-2, -1)的形状为(batch_size, num_head, sql_len+1, head_features),
            # 此时torch.stack()操作后的present张量形状为(2, batch_size, num_head, sql_len+1, head_features)。
            '''present张量形状: (2, batch_size, num_head, sql_len+1, head_features),
            即present张量是用来存储此次迭代中的key张量与上一次迭代中的past_key张量(layer_past[0])合并、
            本次迭代的value张量与上一次迭代中的past_value张量(layer_past[1])合并后所得的新的key张量与value张量的.'''
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        '''此时query张量形状为: (batch_size, num_head, 1, head_features)
        key张量的形状为: (batch_size, num_head, head_features, sql_len+1)
        value张量的形状为: (batch_size, num_head, sql_len+1, head_features)'''
        # 若output_attentions为True, 则self._attn()函数返回的attn_outputs列表中的第二个值为注意力分数张量w.
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)

        # 此时self._attn()函数返回的attn_outputs列表中的第一个元素为多头注意力聚合操作结果张量a,
        # a张量的形状为(batch_size, num_head, 1, head_features);
        # 若output_attentions为True, 则此时self._attn()函数返回的attn_outputs列表中的第二个元素为
        # 注意力分数张量w, 其形状为(batch_size, num_head, 1, seq_len + 1).
        a = attn_outputs[0]

        '''此时经过'多头注意力头合并函数self.merge_heads()'后的多头注意力聚合操作结果张量a的形状
        变为(batch_size, 1, all_head_size), 其中 all_head_size 等于 num_head * head_features, head_features=768.
        all_head_size维度的维度数为768,等于n_state,也等于nx, 即all_head_size=n_state=nx=768.'''
        a = self.merge_heads(a)

        # 此处self.c_proj()为Conv1D(n_state, nx)函数(all_head_size=n_state=nx=768), 相当于一个全连接层的作用,
        # 其将此时的多头注意力聚合操作结果张量a的最后一个维度all_head_size由n_state(768)的维度数投影为nx(768)的维度数.
        a = self.c_proj(a)
        a = self.resid_dropout(a)  # 残差dropout层进行正则化操作, 防止过拟合.

        # 此时多头注意力聚合操作结果张量a的形状为(batch_size, 1, all_head_size),
        # 其中 all_head_size 等于 num_head * head_features；all_head_size维度的维度数为768,
        # 等于n_state,也等于nx, 即all_head_size=n_state=nx=n_embed=768.
        outputs = [a, present] + attn_outputs[1:]

        # 此时返回的outputs列表中:
        # <1> 第一个值为多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
        #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w,
        #     其形状为(batch_size, num_head, 1, seq_len + 1).
        return outputs  # a, present, (attentions)
