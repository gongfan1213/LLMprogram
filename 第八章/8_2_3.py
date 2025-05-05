import torch
import numpy as np

from gensim.models import FastText
model = FastText.load("./xiaohua_fasttext_model_jieba.model")

def get_embedding_model(Word2VecModel):
    vocab_list = [word for word in Word2VecModel.wv.key_to_index]  # 存储所有的词语

    word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {}  # 初始化`[word : vector]`字典

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    ## 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = Word2VecModel.wv[word]  # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵

    #这里的word_vector 数据量较大时不好打印
    return word_index, word_vector, embeddings_matrix	#word_index和embeddings_matrix的作用在下文中阐述


word_index, word_vector, embeddings_matrix = get_embedding_model(model)
print(embeddings_matrix.shape)
import torch
embedding = torch.nn.Embedding(num_embeddings= embeddings_matrix.shape[0],embedding_dim=embeddings_matrix.shape[1])
embedding.weight.data.copy_(torch.tensor(embeddings_matrix))
