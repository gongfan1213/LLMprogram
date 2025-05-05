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
model.eval()
model.load_state_dict(torch.load("./saver/model.pth"))

pinyin_tokens_ids,hanzi_tokens_ids = get_data.get_dataset()
vocab = get_data.vocab
"----------------------------下面是predicate的部分---------------------------------------------------------"

start = 128
end = 256
input_ids = pinyin_tokens_ids[start:end]
batch_predicate = torch.tensor(input_ids).int().to(device)
pred = model(batch_predicate)
pred = torch.softmax(pred,dim=-1)
pred = torch.argmax(pred,dim=-1)

for pinyin,hanzi in zip(input_ids,pred):
    print([vocab[py] for py in pinyin])
    print([vocab[hz] for hz in hanzi])
    print("-----------------------")

