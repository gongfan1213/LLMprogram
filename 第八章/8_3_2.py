import torch
import einops.layers.torch as elt

def char_CNN(input_dim = 28):
    model = torch.nn.Sequential(
        #第一层卷积
        elt.Rearrange("b l c -> b c l"),
        torch.nn.Conv1d(input_dim,32,kernel_size=3,padding=1),
        elt.Rearrange("b c l -> b l c"),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(32),

        #第二层卷积
        elt.Rearrange("b l c -> b c l"),
        torch.nn.Conv1d(32, 28, kernel_size=3, padding=1),
        elt.Rearrange("b c l -> b l c"),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(28),

        #flatten
        torch.nn.Flatten(),  #[batch_size,64 * 28]
        torch.nn.Linear(64 * 28,64),
        torch.nn.ReLU(),

        torch.nn.Linear(64,5),
        torch.nn.Softmax()
    )

    return model

"---------------下面是模型训练部分-----------------------------"
import get_data
from sklearn.model_selection import train_test_split

train_dataset,label_dataset = get_data.get_dataset()
X_train,X_test, y_train, y_test = train_test_split(train_dataset,label_dataset,test_size=0.1, random_state=828)  #将数据集划分为训练集和测试集

#获取device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = char_CNN().to(device)

# 定义交叉熵损失函数
def cross_entropy(pred, label):
    res = -torch.sum(label * torch.log(pred)) / label.shape[0]
    return torch.mean(res)



optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 128
train_num = len(X_test)//128
for epoch in range(99):
    train_loss = 0.
    for i in range(train_num):
        start = i * batch_size
        end = (i + 1) * batch_size

        x_batch = torch.tensor(X_train[start:end]).type(torch.float32).to(device)
        y_batch = torch.tensor(y_train[start:end]).type(torch.float32).to(device)

        pred = model(x_batch)
        loss = cross_entropy(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # 记录每个批次的损失值

    # 计算并打印损失值
    train_loss /= train_num
    accuracy = (pred.argmax(1) == y_batch.argmax(1)).type(torch.float32).sum().item() / batch_size
    print("epoch：",epoch,"train_loss:", round(train_loss,2),"accuracy:",round(accuracy,2))
