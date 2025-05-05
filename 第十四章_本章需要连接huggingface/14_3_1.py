from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
text_generator = TextGenerationPipeline(model, tokenizer)
result = text_generator("从前有座山", max_length=100, do_sample=True)
print(result)



import torch

#注意GPT2LMHeadModel 与 GPT2Model这2个模型，分别接了后面的输出层与没有接输出层的内容
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

print(model)

import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
print(model)
#下面这个代码演示了如何获取某一个层的参数
lm_weight = (model.lm_head.state_dict()["weight"])
torch.save(lm_weight,"./dataset/lm_weight.pth")

from transformers import BertTokenizer, GPT2Model, TextGenerationPipeline
model = GPT2Model.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
import torch
from torch.nn.parameter import Parameter

from transformers import BertTokenizer, GPT2Model, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

class GPT2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #with torch.no_grad():
        self.model = GPT2Model.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

        self.lm_head = torch.nn.Linear(768,21128,bias=False)
        weight = torch.load("../dataset/lm_weight.pth")
        self.lm_head.weight = Parameter(weight)

        self.value_layer = torch.nn.Sequential(torch.nn.Linear(768,1),torch.nn.Tanh(),torch.nn.Dropout(0.1))

    def forward(self,token_inputs):

        embedding = self.model(token_inputs)
        embedding = embedding["last_hidden_state"]

        embedding = torch.nn.Dropout(0.1)(embedding)
        logits = self.lm_head(embedding)

        return logits




