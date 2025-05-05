import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import random

import torch

from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

#from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer

config = {
    "model_name": 'uer/gpt2-chinese-cluecorpussmall',
    "steps": 25000,
    "batch_size": 128,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    "lr": 2e-6,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
    "gen_len": 16,
    "save_freq": 5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

# prompt池
prompts = [
     '酒店',
     '周围',
     '位置',
     '前台'
]

# 情感分类模型
senti_tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
senti_model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=pipe_device)

# 文本生成模型
from moudle import model
gpt2_model = model.GPT2(use_rlhf=True)
gpt2_model_ref = model.GPT2(use_rlhf=True)
gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token
gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

# RL Trainer
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)
total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))

import matplotlib.pyplot as plt
image_list = []
for epoch in tqdm(range(total_ppo_epochs)):
    logs, timing = dict(), dict()
    t0 = time.time()

    batch = {
        'tokens': [],
        'query': []
    }
    for _ in range(config['batch_size']):
        random_prompt = random.choice(prompts)                                  # 随机选择一个prompt
        tokens = gpt2_tokenizer.encode(random_prompt)
        batch['tokens'].append(tokens)
        batch['query'].append(random_prompt)
    query_tensors = [torch.tensor(t[:-1]).long().to(device) for t in batch["tokens"]]

    t = time.time()
    response_tensors = []
    for i in range(config['batch_size']):
        gen_len = config['gen_len']
        prompt_token = query_tensors[i].detach().cpu().numpy()

        response = gpt2_model.generate(prompt_token = prompt_token,       # generate()用于直接生成token_id
                                       continue_buildingsample_num=gen_len)
        response_tensors.append(torch.tensor(response[-gen_len:]).to(device))   #这里输出的是一系列的Token，长度为gen_len
    batch['response'] = [gpt2_tokenizer.decode(r) for r in response_tensors]
    timing['time/get_response'] = time.time() - t

    t = time.time()
    texts = [q + r for q,r in zip(batch['query'], batch['response'])]           # 计算正向/负向情感得分
    pipe_outputs = sentiment_pipe(texts)
    rewards = []
    for output in pipe_outputs:
        if output['label'] == 'positive (stars 4 and 5)':
            rewards.append(output['score'])
        elif output['label'] == 'negative (stars 1, 2 and 3)':
            rewards.append(1 - output['score'])
        else:
            raise ValueError(f"错误的推理结果{output['label']}.")
    rewards = torch.tensor(rewards).to(device)                                  # 将正向情感的得分作为生成得分
    timing['time/get_sentiment_preds'] = time.time() - t

    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)          # PPO Update

    mean_reward = torch.mean(rewards).cpu().numpy()
    image_list.append(mean_reward)
    print()
    print(f"epoch {epoch} mean-reward: {mean_reward}",'Random Sample 5 text(s) of model output:')
    for i in range(5):                                                           # 随机打5个生成的结果
        print(f'{i+1}. {random.choice(texts)}')

print(image_list)
plt.plot(image_list)
plt.show()

torch.save(gpt2_model.state_dict(),"./checkpoints/gpt2_model.pth")
torch.save(gpt2_model_ref.state_dict(),"./checkpoints/gpt2_model_ref.pth")
