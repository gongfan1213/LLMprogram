from transformers import AutoTokenizer, AutoModel
names = ["THUDM/chatglm-6b-int4","THUDM/chatglm-6b"]
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",trust_remote_code=True).half().cuda()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
print("-----------------------")
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)


content="""ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，
基于 General Language Model (GLM) 架构，具有 62 亿参数。
手机号 18888888888
结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 
ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。
邮箱 123456789@qq.com
经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，
账号:root 密码:xiaohua123
62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考我们的博客。
"""
prompt='从上文中，提取"信息"(keyword,content)，包括:"手机号"、"邮箱"、"账号"、"密码"等类型的实体，输出json格式内容'
input ='{}\n\n{}'.format(content,prompt)
print(input)
response, history = model.chat(tokenizer, input, history=[])
print(response)