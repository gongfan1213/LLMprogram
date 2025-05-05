import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

from moudle import model

gpt_model = model.GPT2()
rlhf_weight = torch.load("./RLHF/checkpoints/gpt2_model.pth")
gpt_model.load_state_dict(rlhf_weight)
gpt_model.to("cuda")
inputs_text = "酒店"
input_ids = tokenizer.encode(inputs_text)

for _ in range(10):
    prompt_token = gpt_model.generate(20,prompt_token=input_ids)
    result = tokenizer.decode(prompt_token, skip_special_tokens=True)
    print(result)