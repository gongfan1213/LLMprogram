import torch
import math
import einops.layers.torch as elt


seq = torch.tensor([1,3,4,5,0,0,0])
mask = torch.not_equal(seq,0).float()
embedding = torch.rand(size=(1,7,12))
mask = torch.unsqueeze(mask,dim=-1)
print(embedding * mask)


def create_padding_mark(seq):
    mask = torch.not_equal(seq, 0).float()
    mask = torch.unsqueeze(mask, dim=-1)
    return mask

embedding = torch.rand(size=(5,80,312))
print(torch.nn.LayerNorm(normalized_shape=[80,312])(embedding).shape)






