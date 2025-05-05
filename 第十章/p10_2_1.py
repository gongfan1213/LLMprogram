import torch

class FeedForWard(torch.nn.Module):
    def __init__(self,embdding_dim = 312,scale = 4):
        super().__init__()
        self.linear1 = torch.nn.Linear(embdding_dim,embdding_dim*scale)
        self.relu_1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(embdding_dim*scale,embdding_dim)
        self.relu_2 = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embdding_dim)
    def forward(self,tensor):
        embedding = self.linear1(tensor)
        embedding = self.relu_1(embedding)
        embedding = self.linear2(embedding)
        embedding = self.relu_2(embedding)
        embedding = self.layer_norm(embedding)
        return embedding



if __name__ == '__main__':
    embedding = torch.randn(size=(3,16,312))
    FeedForWard()(embedding)



