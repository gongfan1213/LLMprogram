import torch
from torch.nn.parameter import Parameter

from transformers import BertTokenizer, GPT2Model
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

class GPT2(torch.nn.Module):
    def __init__(self,use_rlhf = False):
        super().__init__()

        self.use_rlhf = use_rlhf
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

        if not self.use_rlhf:
            return logits
        else:
            output = embedding
            value = self.value_layer(output)
            value = torch.squeeze(value,dim=-1)
            return logits,value

    @torch.no_grad()
    def generate(self, continue_buildingsample_num, prompt_token=None, temperature=1., top_p=0.95):
        """
        :param continue_buildingsample_num: 这个的参数指的在输入的prompt_token后再输出多少个字符，
        :param prompt_token: 这个是需要转化成token的内容,这里需要输入的是一个list
        :param temperature:
        :param top_k:
        :return: 输出一个token序列
        """

        prompt_token_new = list(prompt_token)  #使用这个代码，在生成的token里面有102这个分隔符
        for i in range(continue_buildingsample_num):
            _token_inp = torch.tensor([prompt_token_new]).to("cuda")
            if self.use_rlhf:
                result, _ = self.forward(_token_inp)
            else:
                result = self.forward(_token_inp)
            logits = result[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p)  # 预设的top_p = 0.95
            next_token = next_token.reshape(-1)
            prompt_token_new.append(next_token.item())
        return prompt_token_new

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token





