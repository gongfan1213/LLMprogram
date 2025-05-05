import numpy as np
from tqdm import tqdm
import torch
import einops.layers.torch as elt
from tqdm import tqdm

import torch
import numpy as np
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

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

class TextSamplerDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq[:-1],full_seq[1:]


    def __len__(self):
        return self.data.size(0) // self.seq_len



# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, data,vocab,max_length = 96 ):
#         self.vocab = vocab
#         data_size, vocab_size = len(data), len(vocab)
#         print('data has %d characters, %d unique.' % (data_size, vocab_size))
#
#         self.vocab_size = vocab_size
#         self.data = data
#         self.max_length = max_length
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # grab a chunk of (block_size + 1) characters from the data
#
#         #chunk = self.data[idx:idx + self.seq_max_length + 1]
#         sentence = self.data[idx]
#         sentence = [char for char in sentence]
#         input_sentence = ["＞"] +  sentence
#         output_sentence = sentence + ["＜"]
#
#         input_token = [self.vocab.index(char) for char in input_sentence]
#         output_token = [self.vocab.index(char) for char in output_sentence]
#
#         input_token = input_token + [0] * (self.max_length - len(input_token))
#         output_token = output_token + [0] * (self.max_length - len(output_token))
#
#         input_token = torch.tensor(input_token)
#         output_token = torch.tensor(output_token)
#         return input_token, output_token


