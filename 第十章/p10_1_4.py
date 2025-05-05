import torch
import einops.layers.torch as elt


# def splite_tensor(tensor,h_head):
#     embedding = elt.Rearrange("b l (h d) -> b l h d",h = h_head)(tensor)
#     embedding = elt.Rearrange("b l h d -> b h l d", h=h_head)(embedding)
#     return embedding


class Attention(torch.nn.Module):
    def __init__(self,embedding_dim = 312,hidden_dim = 312,n_head = 6):
        super().__init__()
        self.n_head = n_head
        self.query_layer = torch.nn.Linear(embedding_dim, hidden_dim)
        self.key_layer = torch.nn.Linear(embedding_dim, hidden_dim)
        self.value_layer = torch.nn.Linear(embedding_dim, hidden_dim)


    def forward(self,embedding,mask):
        input_embedding = embedding

        query = self.query_layer(input_embedding)
        key = self.key_layer(input_embedding)
        value = self.value_layer(input_embedding)

        query_splited = self.splite_tensor(query,self.n_head)
        key_splited = self.splite_tensor(key,self.n_head)
        value_splited = self.splite_tensor(value,self.n_head)

        key_splited = elt.Rearrange("b h l d -> b h d l")(key_splited)
        # 计算query与key之间的权重系数
        attention_prob = torch.matmul(query_splited, key_splited)

        # 使用softmax对权重系数进行归一化计算
        attention_prob += mask * -1e5  # 在自注意力权重基础上加上掩模值
        attention_prob = torch.softmax(attention_prob, dim=-1)

        # 计算权重系数与value的值从而获取注意力值
        attention_score = torch.matmul(attention_prob, value_splited)
        attention_score = elt.Rearrange("b h l d -> b l (h d)")(attention_score)

        return (attention_score)

    def splite_tensor(self,tensor,h_head):
        embedding = elt.Rearrange("b l (h d) -> b l h d",h = h_head)(tensor)
        embedding = elt.Rearrange("b l h d -> b h l d", h=h_head)(embedding)
        return embedding

def create_padding_mark(seq):
    mask = torch.not_equal(seq, 0).float()
    mask = torch.unsqueeze(mask, dim=-1)
    mask = torch.unsqueeze(mask, dim=1)
    return mask

if __name__ == '__main__':
    embedding = torch.rand(size=(5,16,312))
    mask = torch.ones((5,1,16,1))
    Attention()(embedding,mask)

