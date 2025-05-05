import torch
import torch.utils.data as Data
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW

# 定义下游任务模型
class ModelSimple(torch.nn.Module):
    def __init__(self, pretrain_model_name = "bert-base-chinese"):
        super().__init__()
        self.pretrain_model = BertModel.from_pretrained(pretrain_model_name)
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids):
        with torch.no_grad():  # 上游的模型不进行梯度更新
            output = self.pretrain_model(input_ids=input_ids)  # input_ids: 编码之后的数字(即token) )
        output = self.fc(output[0][:, 0])  # 取出每个 batch 的第一列作为 CLS, 即 (16, 786)
        output = output.softmax(dim=1)  # 通过 softmax 函数, 并使其在 1 的维度上进行缩放，使元素位于[0,1] 范围内，总和为 1
        return output

# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self, pretrain_model_name = "bert-base-chinese"):
        super().__init__()
        self.pretrain_model = BertModel.from_pretrained(pretrain_model_name)
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids,attention_mask,token_type_ids):
        with torch.no_grad():  # 上游的模型不进行梯度更新
            output = self.pretrain_model(input_ids=input_ids,  # input_ids: 编码之后的数字(即token)
                                         attention_mask=attention_mask,  # attention_mask: 其中 pad 的位置是 0 , 其他位置是 1
                                         # token_type_ids: 第一个句子和特殊符号的位置是 0 , 第二个句子的位置是 1
                                         token_type_ids=token_type_ids)
        output = self.fc(output[0][:, 0])  # 取出每个 batch 的第一列作为 CLS, 即 (16, 786)
        output = output.softmax(dim=1)  # 通过 softmax 函数, 并使其在 1 的维度上进行缩放，使元素位于[0,1] 范围内，总和为 1
        return output


