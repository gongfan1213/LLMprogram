from transformers import BertTokenizer, GPT2LMHeadModel,TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem")
text_generator = TextGenerationPipeline(model, tokenizer)
result = text_generator("[CLS] 万 叠 春 山 积 雨 晴 ,", max_length=50, do_sample=True)
print(result)
