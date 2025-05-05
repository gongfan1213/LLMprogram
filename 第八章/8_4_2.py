import torch

import torch
import einops.layers.torch as elt

def word2vec_CNN(input_dim = 28):
    model = torch.nn.Sequential(

        elt.Rearrange("b l d 1 -> b 1 l d"),
        #第一层卷积
        torch.nn.Conv2d(1,3,kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(num_features=3),

        #第二层卷积
        torch.nn.Conv2d(3, 5, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(num_features=5),

        #flatten
        torch.nn.Flatten(),  #[batch_size,64 * 28]
        torch.nn.Linear(2400,64),
        torch.nn.ReLU(),

        torch.nn.Linear(64,5),
        torch.nn.Softmax()
    )

    return model



if __name__ == '__main__':
    image = torch.rand(size=(5,12,64,1))
    model = word2vec_CNN()
    result = model(image)
    print(result.shape)

