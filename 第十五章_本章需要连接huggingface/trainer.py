import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
max_length = 128 + 1
batch_size = 12

device = "cuda"
from moudle import model

save_path = "./train_model_emo.pth"
glm_model = model.GPT2(use_rlhf=False)
glm_model.to(device)
#glm_model.load_state_dict(torch.load(save_path),strict=False)
optimizer = torch.optim.AdamW(glm_model.parameters(), lr=2e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 1200,eta_min=2e-6,last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss()

import get_data_emotion
train_dataset = get_data_emotion.TextSamplerDataset(get_data_emotion.token_list,max_length)
loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)

for epoch in range(30):
    pbar = tqdm(loader, total=len(loader))
    for token_inp,token_tgt in pbar:
        token_inp = token_inp.to(device)
        token_tgt = token_tgt.to(device)

        logits = glm_model(token_inp)
        loss = criterion(logits.view(-1,logits.size(-1)),token_tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # 执行优化器
        pbar.set_description(f"epoch:{epoch +1}, train_loss:{loss.item():.5f}, lr:{lr_scheduler.get_last_lr()[0]*100:.5f}")
    if (epoch + 1) % 2 == 0:
        torch.save(glm_model.state_dict(),save_path)