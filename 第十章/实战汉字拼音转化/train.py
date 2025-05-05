import numpy as np
import torch
import bert
import get_data
max_length = 64
from tqdm import tqdm
vocab_size = get_data.vocab_size
vocab = get_data.vocab

def get_model(embedding_dim = 768):
    model = torch.nn.Sequential(
        bert.BERT(vocab_size=vocab_size),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(embedding_dim,vocab_size)
    )
    return model


device = "cuda"
model = get_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 2400,eta_min=2e-6,last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss()

pinyin_tokens_ids,hanzi_tokens_ids = get_data.get_dataset()
#print(pinyin_tokens_ids)
#print(hanzi_tokens_ids)
class TextSamplerDataset(torch.utils.data.Dataset):
    def __init__(self, pinyin_tokens_ids, hanzi_tokens_ids):
        super().__init__()
        self.pinyin_tokens_ids = pinyin_tokens_ids
        self.hanzi_tokens_ids = hanzi_tokens_ids

    def __getitem__(self, index):
	    #下面的写法是为了遵守GPT2数据输入输出格式而特定的写法
        return torch.tensor(self.pinyin_tokens_ids[index]),torch.tensor(self.hanzi_tokens_ids[index])

    def __len__(self):
        return len(pinyin_tokens_ids)

#model.load_state_dict(torch.load("./saver/model.pth"))
batch_size = 64 #显存太大，这个数可以调大，32,64，128，256，512
from torch.utils.data import DataLoader
loader = DataLoader(TextSamplerDataset(pinyin_tokens_ids,hanzi_tokens_ids),batch_size=batch_size,shuffle=False)


for epoch in range(21):
    pbar = tqdm(loader, total=len(loader))
    for pinyin_inp, hanzi_inp in pbar:

        token_inp = (pinyin_inp).to(device)
        token_tgt = (hanzi_inp).to(device)

        logits = model(token_inp)
        loss = criterion(logits.view(-1,logits.size(-1)),token_tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # 执行优化器
        pbar.set_description(
            f"epoch:{epoch + 1}, train_loss:{loss.item():.5f}, lr:{lr_scheduler.get_last_lr()[0] * 100:.5f}")
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), "./saver/model.pth")

torch.save(model.state_dict(), "./saver/model.pth")




