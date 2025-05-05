
kg_vector_stores = {
    '大规模金融研报': './dataset/financial_research_reports',
    '初始化': './dataset/cache',
}  # 可以替换成自己的知识库，如果没有需要设置为None

from transformers import BertTokenizer, BertModel
import torch
embedding_model_name = "shibing624/text2vec-base-chinese"
embedding_model_length = 512    #这里是由于即使任何长度的文本，都有一个长度限制
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
model = BertModel.from_pretrained(embedding_model_name)

sentences = []
#这里先对每一个文档读取
import os
path = "./dataset/financial_research_reports/"
filelist = [path + i for i in os.listdir(path)]
for file in filelist:
    if file.endswith (".txt"):
        with open(file,"r",encoding="utf-8") as f:
            lines = f.readlines()
            lines = "".join(lines)  #注意这里要做成一个文本
            sentences.append(lines[:embedding_model_length])

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling.

sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings:")
print(sentence_embeddings)
print(sentence_embeddings.shape)

import numpy as np
sentence_embeddings_np = sentence_embeddings.detach().numpy()
np.save("./dataset/financial_research_reports/sentence_embeddings_np.npy",sentence_embeddings_np)







