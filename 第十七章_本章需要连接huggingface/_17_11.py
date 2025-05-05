#from transformers import BertTokenizer, GPT2Model
import utils

prompt_text = "明天是什么天气"
context_list = ["哪个颜色好看","今天晚上吃什么","你家电话多少","明天的天气是晴天","晚上的月亮好美呀"]

sim_results = utils.get_top_n_sim_text(query=prompt_text,documents=context_list,top_n=1)
print(sim_results)










