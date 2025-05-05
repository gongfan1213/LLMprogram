

import torch
from transformers import BertTokenizer
from transformers import BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# pretrain_model = BertModel.from_pretrained("bert-base-chinese")
#
# tokens = tokenizer.encode("春眠不觉晓",max_length=12,padding="max_length",truncation=True)
# print(tokens)
# print("----------------------")
# print(tokenizer("春眠不觉晓",max_length=12,padding="max_length",truncation=True))
# print("----------------------")
#
# tokens = torch.tensor([tokens]).int()
# print(pretrain_model(tokens))



from transformers import BertTokenizer,GPT2Model
model_name = "uer/gpt2-chinese-ancient"
tokenizer = BertTokenizer.from_pretrained(model_name)
pretrain_model = GPT2Model.from_pretrained(model_name)

tokens = tokenizer.encode("春眠不觉晓",max_length=12,padding="max_length",truncation=True)
print(tokens)
print("----------------------")
print(tokenizer("春眠不觉晓",max_length=12,padding="max_length",truncation=True))
print("----------------------")

tokens = torch.tensor([tokens]).int()
print(pretrain_model(tokens))
