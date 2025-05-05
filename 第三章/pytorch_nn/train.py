import numpy as np


def get_one_hot(targets, nb_classes=10):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


train_x = np.load("../../dataset/mnist/x_train.npy");
train_x = np.reshape(train_x, [60000, 784])
train_y = get_one_hot(np.load("../../dataset/mnist/y_train_label.npy"))

import net, model, layer, loss, optimizer

net = net.Net([
    layer.Dense(200),
    layer.ReLU(),
    layer.Dense(100),
    layer.ReLU(),
    layer.Dense(70),
    layer.ReLU(),
    layer.Dense(30),
    layer.ReLU(),
    layer.Dense(10)
])

model = model.Model(net=net, loss=loss.SoftmaxCrossEntropy(), optimizer=optimizer.Adam(lr=2e-4))

loss_list = list()
train_num = 60000 // 128
for epoch in range(20):
    train_loss = 0
    for i in range(train_num):
        start = i * 128
        end = (i + 1) * 128

        inputs = train_x[start:end]
        targets = train_y[start:end]

        pred = model.forward(inputs)
        loss, grads = model.backward(pred,targets)
        model.apply_grads(grads)


        if (i + 1) %50 == 0:
            test_pred = model.forward(inputs)
            test_pred_idx = np.argmax(test_pred, axis=1)

            real_pred_idx = np.argmax(targets, axis=1)

            counter = 0
            for pre,rel in zip(test_pred_idx,real_pred_idx):
                if pre == rel:
                    counter += 1
            print("train_loss:", round(loss, 2), "accuracy:", round(counter/128., 2))

