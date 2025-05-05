import torch
import resnet
import get_data
import numpy as np

train_dataset, label_dataset, test_dataset, test_label_dataset = get_data.get_CIFAR10_dataset(root="../dataset/cifar-10-batches-py/")

train_dataset = np.reshape(train_dataset,[len(train_dataset),3,32,32]).astype(np.float32)/255.
test_dataset = np.reshape(test_dataset,[len(test_dataset),3,32,32]).astype(np.float32)/255.
label_dataset = np.array(label_dataset)
test_label_dataset = np.array(test_label_dataset)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet.resnet18()                  #导入Unet模型
model = model.to(device)                #将计算模型传入GPU硬件等待计算
#model = torch.compile(model)            #Pytorch2.0的特性，加速计算速度
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)   #设定优化函数
loss_fn = torch.nn.CrossEntropyLoss()

batch_size = 64
train_num = len(label_dataset)//batch_size
for epoch in range(60):

    train_loss = 0.
    for i in range(train_num):
        start = i * batch_size
        end = (i + 1) * batch_size

        x_batch = torch.from_numpy(train_dataset[start:end]).to(device)
        y_batch = torch.from_numpy(label_dataset[start:end]).to(device)

        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # 记录每个批次的损失值

    # 计算并打印损失值
    train_loss /= train_num
    accuracy = (pred.argmax(1) == y_batch).type(torch.float32).sum().item() / batch_size

    with torch.no_grad():
        test_num = 2048
        x_test = torch.from_numpy(test_dataset[:test_num]).to(device)
        y_test = torch.from_numpy(test_label_dataset[:test_num]).to(device)
        pred = model(x_test)
        test_accuracy = (pred.argmax(1) == y_test).type(torch.float32).sum().item() / test_num
        print("epoch：",epoch,"train_loss:", round(train_loss,2),";accuracy:",round(accuracy,2),";test_accuracy:",round(test_accuracy,2))


