import numpy as np
sentences = []

src_vocab = {'⊙': 0, '＞': 1, '＜': 2}    #这个是汉字vccab
tgt_vocab = {'⊙': 0, '＞': 1, '＜': 2}    #这个是拼音vocab

with open("../dataset/zh.tsv", errors="ignore", encoding="UTF-8") as f:
    context = f.readlines()
    for line in context:
        line = line.strip().split("	")
        pinyin = line[1]
        hanzi = line[2]
        (hanzi_s) = hanzi.split(" ")
        (pinyin_s) = pinyin.split(" ")
        #[＞＜]
        pinyin_inp = ["＞"] + pinyin_s
        pinyin_trg = pinyin_s + ["＜"]
        line = [hanzi_s,pinyin_inp,pinyin_trg]
        for char in hanzi_s:
            if char not in src_vocab:
                src_vocab[char] = len(src_vocab)
        for char in pinyin_s:
            if char not in tgt_vocab:
                tgt_vocab[char] = len(tgt_vocab)

        sentences.append(line)


# print(src_vocab)
# print(tgt_vocab)
# print(len(src_vocab))
# print(len(tgt_vocab))
#sentences = np.array(sentences)


