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
        weight = torch.load("./dataset/lm_weight.pth")
        self.lm_head.weight = Parameter(weight)

        self.value_layer = torch.nn.Sequential(torch.nn.Linear(768,1),torch.nn.Tanh(),torch.nn.Dropout(0.1))

    def forward(self,token_inputs):

        embedding = self.model(token_inputs)
        embedding = embedding["last_hidden_state"]

        embedding = torch.nn.Dropout(0.1)(embedding)
        logits = self.lm_head(embedding)

        return logits
