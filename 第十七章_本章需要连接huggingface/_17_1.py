#from transformers import BertTokenizer, GPT2Model
import utils

context_list = []

import json
# 打开文件,r是读取,encoding是指定编码格式
with open('./dataset/train1.json', 'r', encoding='utf-8') as fp:
    # load()函数将fp(一个支持.read()的文件类对象，包含一个JSON文档)反序列化为一个Python对象
    data = json.load(fp)
    for line in data:
        line = (line["context_text"])
        context_list.append(line)

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",trust_remote_code=True).half().cuda()

prompt_text = "小孩牙龈肿痛服用什么药"
"-------------------------------------------------------------------------------------------------------------------------------------------------------"
print("普通chatGLM询问结果：")
response, _ = model.chat(tokenizer, prompt_text, history=[])
print(response)
print("----------------------------------------")
print("下面是经过文本查询的结果如下所示：")
sim_results = utils.get_top_n_sim_text(query=prompt_text,documents=context_list)
print(sim_results)
print("----------------------------------------")
print("由chatGLM根据文档的结果如下：")
prompt = utils.generate_prompt(prompt_text,sim_results)
response, _ = model.chat(tokenizer, prompt, history=[])
print(response)
print("----------------------------------------")




# tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# model = GPT2Model.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
#
#
# class GLMBot:
#     def __init__(self):
#         super().__init__()






