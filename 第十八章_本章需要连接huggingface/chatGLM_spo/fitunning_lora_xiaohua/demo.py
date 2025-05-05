import torch

from transformers import AutoTokenizer
from torch.utils.data import RandomSampler, DataLoader
from 第十八章_本章需要连接huggingface.chatGLM_spo.huggingface_saver import xiaohua_model,configuration_chatglm,modeling_chatglm
from tqdm import tqdm
config = configuration_chatglm.ChatGLMConfig()
#这里是设置config中的pre_seq_len  与 prefix_projection ，只有这2个设置好了才行

model = xiaohua_model.XiaohuaModel(model_path="../huggingface_saver/chatglm6b.pth",config=config,strict=False)
model = model.half().cuda()

for name,param in model.named_parameters():
    param.requires_grad = False

# 下面就是lora的部分
from minlora.model import  *
from minlora.utils import *

for key,_layer in model.named_modules():
    if "query_key_value" in key:
        add_lora(_layer)
xiaohua_model.print_trainable_parameters(model)

prompt_text = "按给定的格式抽取文本信息。\n文本:"#"你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
from 第十八章_本章需要连接huggingface.chatGLM_spo import get_data
all_train_data = get_data.get_train_data("../data/spo_0.json",tokenizer, 32, 48, prompt_text)
train_dataset = get_data.Seq2SeqDataSet(all_train_data)
train_loader = DataLoader(train_dataset,  batch_size=2,  drop_last=True,collate_fn=get_data.coll_fn, num_workers=0)

from accelerate import Accelerator
accelerator = Accelerator()

lora_parameters = [{"params": list(get_lora_params(model))}]
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2400, eta_min=2e-6, last_epoch=-1)
model, optim, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
for epoch in range(96):
    pbar = tqdm(train_loader, total=len(train_loader))
    for batch in (pbar):
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        _,_,loss = model.forward(input_ids,labels=labels)
        accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        lr_scheduler.step()  # 执行优化器
        optimizer.zero_grad()

        pbar.set_description(
            f"epoch:{epoch + 1}, train_loss:{loss.item():.5f}, lr:{lr_scheduler.get_last_lr()[0] * 1000:.5f}")

torch.save(model.state_dict(), "./glm6b_lora_all.pth")
lora_state_dict = get_lora_state_dict(model)
torch.save(lora_state_dict, "./glm6b_lora_only.pth")

