import csv
import numpy as np
import re

def text_clearTitle_word2vec(text):
    text = text.lower()             #将文本转化成小写
    text = re.sub(r"[^a-z]"," ",text)   #替换非标准字符，^是求反操作。
    text = re.sub(r" +", " ", text)     #替换多重空格
    text = text.strip()              #取出首尾空格
    text = text + " eos"          	#添加结束符，注意eos前有空格
    text = text.split(" ")			#对文本分词转成列表存储
    return text

#生成标题的one-hot标签
def get_label_one_hot(list):
    values = np.array(list)
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]

def get_word2vec_dataset(n = 12):
    agnews_label = []      					#创建标签列表
    agnews_title = []       					#创建标题列表
    agnews_train = csv.reader(open("../dataset/ag_news数据集/dataset/train.csv","r"))
    for line in agnews_train:    				#将数据读取对应列表中
        agnews_label.append(int(line[0]))
        agnews_title.append(text_clearTitle_word2vec(line[1]))   #先将数据进行清洗之后再读取
    from gensim.models import word2vec  		# 导入gensim包
    model = word2vec.Word2Vec(agnews_title, vector_size=64, min_count=0, window=5)  # 设置训练参数
    train_dataset = []     					#创建训练集列表
    for line in agnews_title:       				#对长度进行判定
        length = len(line)        				#获取列表长度
        if length > n:            				#对列表长度进行判断
            line = line[:n]       				#截取需要的长度列表
            word2vec_matrix = (model.wv[line])   	#获取word2vec矩阵
            train_dataset.append(word2vec_matrix)    #将word2vec矩阵添加到训练集中
        else:                  #补全长度不够的操作
            word2vec_matrix = (model.wv[line])     	#获取word2vec矩阵
            pad_length = n - length             	#获取需要补全的长度
            pad_matrix = np.zeros([pad_length, 64]) + 1e-10    #创建补全矩阵并增加一个小数值
            word2vec_matrix = np.concatenate([word2vec_matrix, pad_matrix], axis=0) #矩阵补全
            train_dataset.append(word2vec_matrix)          #将word2vec矩阵添加到训练集中
    train_dataset = np.expand_dims(train_dataset,3)  		 #将三维矩阵进行扩展
    label_dataset = get_label_one_hot(agnews_label)    	 #转换成onehot矩阵
    return train_dataset, label_dataset

if __name__ == '__main__':

    train_dataset, label_dataset = get_word2vec_dataset()
    print(train_dataset.shape)
    print(label_dataset.shape)