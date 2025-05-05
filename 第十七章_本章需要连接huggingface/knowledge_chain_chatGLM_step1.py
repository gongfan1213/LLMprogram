import numpy
from utils import mean_pooling,compute_sim_score


from transformers import BertTokenizer, BertModel
import torch
embedding_model_name = "shibing624/text2vec-base-chinese"
# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
model = BertModel.from_pretrained(embedding_model_name)

query = ["雅生活服务的人工成本占营业成本的比例是多少"]
# Tokenize sentences
query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
# Compute token embeddings
with torch.no_grad():
    model_output = model(**query_input)
# Perform pooling. In this case, mean pooling.
query_embedding = mean_pooling(model_output, query_input['attention_mask'])
print(query_embedding.shape)

import numpy as np
sentence_embeddings_np =  np.load("./dataset/financial_research_reports/sentence_embeddings_np.npy")

#这里很容易认清楚是007
for i in  range(len(sentence_embeddings_np)):
    score = compute_sim_score(sentence_embeddings_np[i],query_embedding[0])
    print(i,score)
