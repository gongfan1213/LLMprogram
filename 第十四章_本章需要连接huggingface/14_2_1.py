#!/usr/bin/env Python
# coding=utf-8

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from _14_1_2 import *
# 初始化GPT2模型的Tokenizer类.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# 初始化GPT2模型, 此处以初始化GPT2LMHeadModel()类的方式调用GPT2模型.
model = GPT2LMHeadModel.from_pretrained('gpt2')
# model.config.use_return_dict = None
# print(model.config.use_return_dict)

# GPT模型第一次迭代的输入的上下文内容, 将其编码以序列化.
# 同时, generated也用来存储GPT2模型所有迭代生成的token索引.
generated = tokenizer.encode("The Manhattan bridge")
# 将序列化后的第一次迭代的上下文内容转化为pytorch中的tensor形式.
context = torch.tensor([generated])
# 第一次迭代时还无past_key_values元组.
past_key_values = None

for i in range(30):

    '''
    此时模型model返回的output为CausalLMOutputWithPastAndCrossAttentions类,
    模型返回的logits以及past_key_values对象为其中的属性,
    CausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
'''
    output = model(context, past_key_values=past_key_values)
    past_key_values = output.past_key_values
    # 此时获取GPT2模型计算的输出结果hidden_states张量中第二维度最后一个元素的argmax值, 得出的argmax值即为此次GPT2模型迭代
    # 计算生成的下一个token. 注意, 此时若是第一次迭代, 输出结果hidden_states张量的形状为(batch_size, sel_len, n_state);
    # 此时若是第二次及之后的迭代, 输出结果hidden_states张量的形状为(batch_size, 1, n_state), all_head_size=n_state=nx=768.
    token = torch.argmax(output.logits[..., -1, :])

    # 将本次迭代生成的token的张量变为二维张量, 以作为下一次GPT2模型迭代计算的上下文context.
    context = token.unsqueeze(0)
    # 将本次迭代计算生成的token的序列索引变为列表存入generated
    generated += [token.tolist()]

# 将generated中所有的token的索引转化为token字符.
sequence = tokenizer.decode(generated)
sequence = sequence.split(".")[:-1]
print(sequence)
