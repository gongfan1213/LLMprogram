
import torch

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

#这里需要注意的是，为了节省显存的原因，我们使用的是half数据格式，也就是半精度的
#使用全精度需要注意可能会重新下载模型存档文件
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
#model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cpu()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
    # #你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

torch.save(model.state_dict(),"./huggingface_saver/chatglm6b.pth")


