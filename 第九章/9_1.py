import numpy as np

max_length = 80         #设置获取的文本长度为80
labels = []             #用以存放label
context = []            #用以存放汉字文本
vocab = set()           #
with open("../dataset/cn/ChnSentiCorp.txt", mode="r", encoding="UTF-8") as emotion_file:
    for line in emotion_file.readlines():
        line = line.strip().split(",")

        # labels.append(int(line[0]))
        if int(line[0]) == 0:
            labels.append(0)    #这里由于我们在后面直接采用Pytorch自带的crossentroy函数，所以这里直接输入0，否则输入[1,0]
        else:
            labels.append(1)
        text = "".join(line[1:])
        context.append(text)
        for char in text: vocab.add(char)   #建立vocab和vocab编号

voacb_list = list(sorted(vocab))
# print(len(voacb_list))
token_list = []
#下面的内容是对context内容根据vocab进行token处理
for text in context:
    token = [voacb_list.index(char) for char in text]
    token = token[:max_length] + [0] * (max_length - len(token))
    token_list.append(token)


seed = 17
np.random.seed(seed);np.random.shuffle(token_list)
np.random.seed(seed);np.random.shuffle(labels)

dev_list = np.array(token_list[:170])
dev_labels = np.array(labels[:170])

token_list = np.array(token_list[170:])
labels = np.array(labels[170:])

import torch
class RNNModel(torch.nn.Module):
    def __init__(self,vocab_size = 128):
        super().__init__()
        self.embedding_table = torch.nn.Embedding(vocab_size,embedding_dim=312)
        self.gru  =  torch.nn.GRU(312,256)  # 注意这里输出两个，out与hidden，这里的out是序列在模型运行后全部隐藏层的状态，而hidden是最后一个隐藏层的状态
        self.batch_norm = torch.nn.LayerNorm(256,256)

        self.gru2  =  torch.nn.GRU(256,128,bidirectional=True)  # 注意这里输出两个，out与hidden，这里的out是序列在模型运行后全部隐藏层的状态，而hidden是最后一个隐藏层的状态


    def forward(self,token):
        token_inputs = token
        embedding = self.embedding_table(token_inputs)
        gru_out,_ = self.gru(embedding)

        embedding = self.batch_norm(gru_out)
        out,hidden = self.gru2(embedding)

        return out


def get_model(vocab_size = len(voacb_list),max_length = max_length):
    model = torch.nn.Sequential(
        RNNModel(vocab_size),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * max_length * 128,2)
    )
    return model




device = "cuda"
model = get_model().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

loss_func = torch.nn.CrossEntropyLoss()


batch_size = 128
train_length = len(labels)
for epoch in (range(21)):
    train_num = train_length // batch_size
    train_loss, train_correct = 0, 0
    for i in (range(train_num)):
        start = i * batch_size
        end = (i + 1) * batch_size

        batch_input_ids = torch.tensor(token_list[start:end]).to(device)
        batch_labels = torch.tensor(labels[start:end]).to(device)

        pred = model(batch_input_ids)

        loss = loss_func(pred, batch_labels.type(torch.uint8))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += ((torch.argmax(pred, dim=-1) == (batch_labels)).type(torch.float).sum().item() / len(batch_labels))

    train_loss /= train_num
    train_correct /= train_num
    print("train_loss:", train_loss, "train_correct:", train_correct)

    test_pred = model(torch.tensor(dev_list).to(device))
    correct = (torch.argmax(test_pred, dim=-1) == (torch.tensor(dev_labels).to(device))).type(torch.float).sum().item() / len(test_pred)
    print("test_acc:",correct)
    print("-------------------")













