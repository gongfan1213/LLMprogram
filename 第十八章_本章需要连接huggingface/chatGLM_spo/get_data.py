
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

def get_train_data(data_path,tokenizer,max_len, max_src_len, prompt_text):
    max_tgt_len = max_len - max_src_len - 3
    all_data = []
    with open(data_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            sample = json.loads(line.strip())
            src_tokens = tokenizer.tokenize(sample["text"])
            prompt_tokens = tokenizer.tokenize(prompt_text)

            if len(src_tokens) > max_src_len - len(prompt_tokens):
                src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]

            tgt_tokens = tokenizer.tokenize("\n原因:"+sample["answer"])

            if len(tgt_tokens) > max_tgt_len:
                tgt_tokens = tgt_tokens[:max_tgt_len]
            tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]

            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            all_data.append(
                {"text": sample["text"], "answer": sample["answer"], "input_ids": input_ids, "labels": labels})
    return all_data

class Seq2SeqDataSet(Dataset):
    """数据处理函数"""
    def __init__(self, all_data):
        # prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def coll_fn(batch):
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=3),   #这里原来是20003，我看了vocab改成了3
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=3)}

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    all_data = get_train_data("./data/spo_0.json",tokenizer, 768, 450, "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：")

    train_dataset = Seq2SeqDataSet(all_data)
    instance = train_dataset.__getitem__(0)
    text,ans,input_ids,lab = instance
    print(len(instance["input_ids"]))
    print(len(instance["labels"]))

    from torch.utils.data import RandomSampler, DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, drop_last=True, num_workers=0)






