import torch
from moudle import model

class GPT_XiaoHua(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt = model.GPT2(use_rlhf=True)

        with torch.no_grad():
            self.gpt_no_grad = model.GPT2(use_rlhf=True)

    def forward(self,token_inputs,need_grad = True):

        if need_grad:
            return self.gpt(token_inputs)

        return self.gpt_no_grad(token_inputs)



