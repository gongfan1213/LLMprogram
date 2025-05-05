text = [
    "卷积神经网络在图像处理领域获得了极大成功，其结合特征提取和目标训练为一体的模型能够最好的利用已有的信息对结果进行反馈训练。",
    "对于文本识别的卷积神经网络来说，同样也是充分利用特征提取时提取的文本特征来计算文本特征权值大小的，归一化处理需要处理的数据。",
    "这样使得原来的文本信息抽象成一个向量化的样本集，之后将样本集和训练好的模板输入卷积神经网络进行处理。",
    "本节将在上一节的基础上使用卷积神经网络实现文本分类的问题，这里将采用两种主要基于字符的和基于word embedding形式的词卷积神经网络处理方法。",
    "实际上无论是基于字符的还是基于word embedding形式的处理方式都是可以相互转换的，这里只介绍使用基本的使用模型和方法，更多的应用还需要读者自行挖掘和设计。"
]

import jieba
import numpy as np

jieba_cut_list = []
for line in text:
    jieba_cut = jieba.lcut(line)
    jieba_cut_list.append(jieba_cut)
    print(jieba_cut)

from gensim.models import FastText
model = FastText(min_count=5,vector_size=300,window=7,workers=10,epochs=50,seed=17,sg=1,hs=1)
model.build_vocab(jieba_cut_list)
model.train(jieba_cut_list, total_examples=model.corpus_count, epochs=model.epochs)#这里使用笔者给出的固定格式即可
model.save("./xiaohua_fasttext_model_jieba.model")

from gensim.models import FastText
model = FastText.load("./xiaohua_fasttext_model_jieba.model")
#embedding = model.wv["卷积"]#卷积与神经网络，这2个词都是经过预训练的。

model.build_vocab(jieba_cut_list, update=True)	# second_sentences是新的训练数据，处理方法和上一样
model.train(jieba_cut_list, total_examples=model.corpus_count, epochs=6)
model.min_count = 10
model.save("./xiaohua_fasttext_model_jieba.model")

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


if __name__ == '__main__':
    get_embedding_model(model)