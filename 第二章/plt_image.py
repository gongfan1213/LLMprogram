import numpy as np
import matplotlib.pyplot as plt
import torch

grid_size = 5
x_train = np.load("../dataset/mnist/x_train.npy")
image = torch.tensor(x_train[0]).to("cuda")
image = image
print(image.shape)
image = image.cpu().numpy()
plt.imshow(image)
plt.savefig("./img/img.jpg")
plt.show()




# image = torch.tensor(x_train[0]).to("cuda")
# image = torch.squeeze(image,dim=0)
# image = image.cpu().numpy()
# plt.imshow(image)
# plt.show()


