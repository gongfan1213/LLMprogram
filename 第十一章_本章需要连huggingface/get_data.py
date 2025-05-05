import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

max_length = 80         #设置获取的文本长度为80
labels = []             #用以存放label
context = []            #用以存放汉字文本
token_list = []

with open("../dataset/cn/ChnSentiCorp.txt", mode="r", encoding="UTF-8") as emotion_file:
    for line in emotion_file.readlines():
        line = line.strip().split(",")

        # labels.append(int(line[0]))
        if int(line[0]) == 0:
            labels.append(0)    #这里由于我们在后面直接采用Pytorch自带的crossentroy函数，所以这里直接输入0，否则输入[1,0]
        else:
            labels.append(1)
        text = "".join(line[1:])
        token = tokenizer.encode(text,max_length=max_length,padding="max_length",truncation=True)

        token_list.append(token)
        context.append(text)

seed = 828
np.random.seed(seed);np.random.shuffle(token_list)
np.random.seed(seed);np.random.shuffle(labels)

dev_list = np.array(token_list[:170]).astype(int)
dev_labels = np.array(labels[:170]).astype(int)

token_list = np.array(token_list[170:]).astype(int)
labels = np.array(labels[170:]).astype(int)
