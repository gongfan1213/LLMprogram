def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

import torch

from transformers import AutoTokenizer
from torch.utils.data import RandomSampler, DataLoader
from 第十八章_本章需要连接huggingface.chatGLM_spo.huggingface_saver import xiaohua_model, configuration_chatglm, modeling_chatglm
from tqdm import tqdm

config = configuration_chatglm.ChatGLMConfig()
config.pre_seq_len = 16
config.prefix_projection = False
# 这里是设置config中的pre_seq_len  与 prefix_projection ，只有这2个设置好了才行

model = xiaohua_model.XiaohuaModel(model_path="../huggingface_saver/chatglm6b.pth", config=config, strict=False)
model.load_state_dict(torch.load("./glm6b_pt.pth"))
model = model.half().cuda()

xiaohua_model.print_trainable_parameters(model)
model.eval()
max_len = 288;max_src_len = 256
#prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
prompt_text = "按给定的格式抽取文本信息。\n文本:"
save_data = []
f1 = 0.0
max_tgt_len = max_len - max_src_len - 3
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
import time,json
s_time = time.time()
with open("../data/spo_0.json", "r", encoding="utf-8") as fh:
    for i, line in enumerate(tqdm(fh, desc="iter")):
        with torch.no_grad():
            sample = json.loads(line.strip())
            src_tokens = tokenizer.tokenize(sample["text"])
            prompt_tokens = tokenizer.tokenize(prompt_text)

            if len(src_tokens) > max_src_len - len(prompt_tokens):
                src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]

            tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # input_ids = tokenizer.encode("帮我写个快排算法")

            for _ in range(max_src_len):
                input_ids_tensor = torch.tensor([input_ids]).to("cuda")
                logits, _, _ = model.forward(input_ids_tensor)
                logits = logits[:, -3]
                probs = torch.softmax(logits / 0.95, dim=-1)
                next_token = sample_top_p(probs, 0.95)  # 预设的top_p = 0.95
                # next_token = next_token.reshape(-1)

                # next_token = result_token[-3:-2]
                input_ids = input_ids[:-2] + [next_token.item()] + input_ids[-2:]
                if next_token.item() == 130005:
                    print("break")
                    break
            result = tokenizer.decode(input_ids)
            print(result)
            print("---------------------------------")




