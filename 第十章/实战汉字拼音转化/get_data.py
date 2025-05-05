
from tqdm import  tqdm
pinyin_list = [];hanzi_list = []
vocab = set()

max_length = 64

with open("zh.tsv", errors="ignore", encoding="UTF-8") as f:
    context = f.readlines()
    for line in context:
        line = line.strip().split("	")
        pinyin = line[1].split(" ");hanzi = line[2].split(" ")
        for _pinyin, _hanzi in zip(pinyin, hanzi):
            vocab.add(_pinyin);            vocab.add(_hanzi);

        pinyin = pinyin + ["PAD"] * (max_length - len(pinyin))
        hanzi = hanzi + ["PAD"] * (max_length - len(hanzi))
        if len(pinyin) <= max_length:
            pinyin_list.append(pinyin);hanzi_list.append(hanzi)

vocab = ["PAD"] + list(sorted(vocab))
vocab_size = len(vocab)

#这里截取一部分数据
pinyin_list = pinyin_list[:3000]
hanzi_list = hanzi_list[:3000]


def get_dataset():
    pinyin_tokens_ids = []
    hanzi_tokens_ids = []

    for pinyin,hanzi in zip(tqdm(pinyin_list),hanzi_list):
        pinyin_tokens_ids.append([vocab.index(char) for char in pinyin])
        hanzi_tokens_ids.append([vocab.index(char) for char in hanzi])
    #len(pinyin_vocab): 1154
    #len(hanzi_vocab): 4462
    return pinyin_tokens_ids,hanzi_tokens_ids



if __name__ == '__main__':

    pinyin_tokens_ids,hanzi_tokens_ids = get_dataset()

    for i in range(1024):
        pinyin = (pinyin_tokens_ids[i])
        hanzi = (hanzi_tokens_ids[i])
        print([vocab[py] for py in pinyin])
        print([vocab[hz] for hz in hanzi])
        print("------------------")