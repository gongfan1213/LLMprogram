import torch
from _14_1_2 import *
class Block(torch.nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        # config对应的GPT2Config()类中, n_embd属性默认为768, 因此此处hidden_size即为768.
        hidden_size = config.n_embd
        # config对应的GPT2Config()类中, n_inner属性默认为None, 因此此处inner_dim一般都为4 * hidden_size.
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 此处n_ctx即等于config对应的GPT2Config()类中的n_ctx属性, 其值为1024.
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

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

        '''
        <1> 此时的隐藏状态hidden_states的形状为 (batch_size, 1, nx), 此时nx = n_state = n_embed = all_head_size = 768，
            即此时隐藏状态hidden_states的形状为(batch_size, 1, 768)。
        <2> 此时layer_past为一个存储着past_key张量与past_value张量的大张量, 其
             形状为(2, batch_size, num_head, sql_len, head_features).
        <3> attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,
             其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.
        '''

        # 将此时输入的隐藏状态hidden_states先输入进LayerNormalization层进行层标准化计算后,
        # 再将标准化结果输入进'多头注意力计算层self.attn()'中进行多头注意力聚合操作计算.
        # 此时返回的attn_outputs列表中:
        # <1> 第一个值为多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
        #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # 此时的attn_output张量为返回的attn_outputs列表中第一个值:
        # 多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        attn_output = attn_outputs[0]  # output_attn列表: a, present, (attentions)
        outputs = attn_outputs[1:]

        # residual connection, 进行残差连接.
        # 此时attn_output张量形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        # hidden_states的形状为(batch_size, 1, 768).
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            '''此时self.crossattention()的Cross_Attention运算过程与self.attn()的Attention运算过程几乎相同, 其不同点在于：

            <1> self.attn()的Attention运算是将LayerNormalization之后的hidden_states通过'self.c_attn = Conv1D(3 * n_state, nx)
            (第165行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 3 * 768), 再将投影后的hidden_states
            在第三维度(dim=2)上拆分为三份分别赋为query、key、value, 其形状都为(batch_size, 1, 768)；
            此时n_state = nx = num_head*head_features = 768.

            之后经过split_heads()函数拆分注意力头且key、value张量分别与past_key、past_value张量合并之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, sql_len+1),
            value张量的形状变为(batch_size, num_head, sql_len+1, head_features).

            <2> self.crossattention()的Cross_Attention运算过程则是将LayerNormalization之后的hidden_states通过
            'self.q_attn = Conv1D(n_state, nx)(第163行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 768),
            将此投影之后的hidden_states赋值作为query张量；
            再将此时从编码器(encoder)中传过来的编码器隐藏状态encoder_hidden_states通过'self.c_attn = Conv1D(2 * n_state, nx)
            (第162行代码)'将encoder_hidden_states的形状由(batch_size, enc_seq_len, 768)投影为(batch_size, enc_seq_len, 2 * 768),
            将投影后的encoder_hidden_states在在第三维度(dim=2)上拆分为两份分别赋为key、value,
            其形状都为(batch_size, enc_seq_len, 768)； 此时n_state = nx = num_head*head_features = 768.

            之后经过split_heads()函数拆分注意力头之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, enc_seq_len),
            value张量的形状变为(batch_size, num_head, enc_seq_len, head_features).
            此时计算出的cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).'''

            # 此时将上方的隐藏状态hidden_states(Attention运算结果+Attention运算前的hidden_states)先输入进LayerNormalization
            # 层进行层标准化计算后, 再将标准化结果输入进'交叉多头注意力计算层self.crossattention()'中与编码器传入的隐藏状态
            # encoder_hidden_states进行交叉多头注意力聚合操作计算.
            # 此时返回的cross_attn_outputs列表中:
            # <1> 第一个值为与编码器传入的隐藏状态encoder_hidden_states进行交叉多头注意力聚合操作的结果张量a,
            #     形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768。
            # <2> 第二个值仍为present张量, 但由于此时是做'交叉多头注意力计算self.crossattention()',此时输入进self.crossattention()
            #     函数的参数中不包含layer_past(来自past_key_values列表)的past_key与past_value张量, 因此此时的present为(None,),
            #     详细代码可见本脚本代码357行, 因此此处用不到'交叉多头注意力计算结果列表cross_attn_outputs'中的present,
            #     将其舍弃(代码第528行)。
            # <3> 若output_attentions为True, 则第三个值为: 交叉注意力分数张量w, 即cross attentions,
            #      cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            # cross_attn_outputs[2:] add cross attentions if we output attention weights,
            # 即将'交叉多头注意力计算结果列表cross_attn_outputs'中的交叉注意力分数张量cross_attention保存为此时的
            # outputs列表中的最后一个元素.
            outputs = outputs + cross_attn_outputs[2:]

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs

        # 此时返回的outputs列表中的元素为：
        # <1> 第一个值为多头注意力聚合操作结果张量hidden_states输入前馈MLP层与残差连接之后得到的最终hidden_states张量,
        #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
        #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
        # <4> 若此时进行了Cross Attention计算, 则第四个值为'交叉多头注意力计算结果列表cross_attn_outputs'中的
        #     交叉注意力分数张量cross_attention, 其形状为(batch_size, num_head, 1, enc_seq_len).
        return outputs  # hidden_states, present, (attentions, cross_attentions)
