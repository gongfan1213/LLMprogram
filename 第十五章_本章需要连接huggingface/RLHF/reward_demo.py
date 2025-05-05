import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

pipe_device = 0 if torch.cuda.is_available() else -1
# 情感分类模型
senti_tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
senti_model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=pipe_device)

text = ["这家店东西很好吃，我但是饮料不怎么样。","这家店的东西很好吃，我很喜欢，推荐！"]

result = sentiment_pipe(text[0])
print(text[0],result)
print("-----------------------------")
result = sentiment_pipe(text[1])
print(text[1],result)