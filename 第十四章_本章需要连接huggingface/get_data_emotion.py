import torch
import numpy as np
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

#首先获取情感分类数据
token_list = []
with open("./dataset/ChnSentiCorp.txt", mode="r", encoding="UTF-8") as emotion_file:
    for line in emotion_file.readlines():
        line = line.strip().split(",")

        text = "".join(line[1:])
        inputs = tokenizer(text,return_tensors='pt')
        token = input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        for id in token[0]:
            token_list.append(id.item())
token_list = torch.tensor(token_list * 5)
#调用标准的数据输入格式
class TextSamplerDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
	    #下面的写法是为了遵守GPT2数据输入输出格式而特定的写法
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq[:-1],full_seq[1:]


    def __len__(self):
        return self.data.size(0) // self.seq_len
