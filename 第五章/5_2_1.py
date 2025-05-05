import torch
import torch.nn as nn
import numpy as np
import einops.layers.torch as elt

#载入数据
x_train = np.load("../dataset/mnist/x_train.npy")
y_train_label = np.load("../dataset/mnist/y_train_label.npy")

x_train = np.expand_dims(x_train,axis=1)
print(x_train.shape)



