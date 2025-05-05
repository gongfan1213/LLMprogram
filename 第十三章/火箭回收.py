
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import gym
import time
import numpy as np
import random
from IPython import display


class Memory:
    def __init__(self):
        """初始化"""
        self.actions = []  # 行动(共4种)
        self.states = []  # 状态, 由8个数字组成
        self.logprobs = []  # 概率
        self.rewards = []  # 奖励
        self.is_dones = []       ## 游戏是否结束 is_terminals?

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_dones[:]



class Action(torch.nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super().__init__()
        # actor
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        action_logits = self.action_layer(state)  # 计算4个方向概率
        return action_logits

class Value(torch.nn.Module):
    def __init__(self, state_dim=8):
        super().__init__()
        # value
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, state):
        state_value = self.value_layer(state)
        return state_value




class PPOAgent:
    def __init__(self,state_dim,action_dim,n_latent_var,lr,betas,gamma, K_epochs, eps_clip):
        self.lr = lr  # 学习率
        self.betas = betas  # betas
        self.gamma = gamma  # gamma
        self.eps_clip = eps_clip  # 裁剪, 限制值范围
        self.K_epochs = K_epochs  # 获取的每批次的数据作为训练使用的次数

        # action
        self.action_layer = Action()
        # critic
        self.value_layer = Value()

        self.optimizer = torch.optim.Adam([{"params":self.action_layer.parameters()},{"params":self.value_layer.parameters()}], lr=lr, betas=betas)

        #损失函数
        self.MseLoss = torch.nn.MSELoss()

    def evaluate(self,state,action):

        action_probs = self.action_layer(state)     #这里输出的结果是一个4类别的东西 [-1,4]
        dist = Categorical(action_probs)    # 转换成类别分布
            # 计算概率密度, log(概率)
        action_logprobs = dist.log_prob(action)
           # 计算信息熵
        dist_entropy = dist.entropy()

            # 评判，对当前的状态进行评判
        state_value = self.value_layer(state)

            # 返回行动概率密度, 评判值, 行动概率熵
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def update(self,memory):
        # 预测状态回报
        rewards = []
        discounted_reward = 0   #discounted = 不重要

        #这里是不是可以这样理解，当前步骤是决定未来的步骤，而模型需要根据当前步骤对未来的最终结果进行修正，如果他遵循了现在的步骤，就可以看到未来的结果如何，
        #而这未来的j结果会很差，所以模型需要远离会造成坏的结果的步骤，所以就
        #所以就反过来计算
        #print(len(self.memory.rewards),len(self.memory.is_dones))  这里就是做成批次，1200批次数据做一次
        for reward, is_done in zip(reversed(memory.rewards), reversed(memory.is_dones)):
            # 回合结束
            if is_done:
                discounted_reward = 0

            # 更新削减奖励(当前状态奖励 + 0.99*上一状态奖励
            discounted_reward = reward + (self.gamma * discounted_reward)
            # 首插入
            rewards.insert(0, discounted_reward)
        #print(len(rewards))        #这里的长度就是根据batch_size的长度设置
        # 标准化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)       #

        #print(len(self.memory.states),len(self.memory.actions),len(self.memory.logprobs))      ##这里的长度就是根据batch_size的长度设置
        ## 张量转换
        # convert list to tensor
        old_states = torch.tensor(memory.states)
        old_actions = torch.tensor(memory.actions)
        old_logprobs = torch.tensor(memory.logprobs)

        #代优化 K 次:
        for _ in range(5):
            # Evaluating old actions and values : 新策略 重用 旧样本进行训练
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)


            ratios =  torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2)  +  0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy


            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def act(self,state):

        state = torch.from_numpy(state).float()
        # 计算4个方向概率
        action_probs = self.action_layer(state)
        # 通过最大概率计算最终行动方向
        dist = Categorical(action_probs)
        action = dist.sample()    #这个是根据action_probs做出符合分布action_probs的抽样结果

        return action.item(),dist.log_prob(action)


state_dim = 8 ### 游戏的状态是个8维向量
action_dim = 4 ### 游戏的输出有4个取值
n_latent_var = 128           # 神经元个数
update_timestep = 1200      # 每1200步policy更新一次
lr = 0.002                  # learning rate
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 5                # policy迭代更新次数
eps_clip = 0.2              # clip parameter for PPO  论文中表明0.2效果不错
random_seed = 929

agent = PPOAgent(state_dim ,action_dim,n_latent_var,lr,betas,gamma,K_epochs,eps_clip)
memory = Memory()
# agent.network.train()  # Switch network into training mode
EPISODE_PER_BATCH = 5  # update the  agent every 5 episode
NUM_BATCH = 200     # totally update the agent for 400 time


avg_total_rewards, avg_final_rewards = [], []

env = gym.make('LunarLander-v2', render_mode='rgb_array')
rewards_list = []
for i in range(200):

    rewards = []
    # collect trajectory
    for episode in range(EPISODE_PER_BATCH):
        ### 重开一把游戏
        state = env.reset()[0]

        while True:
            #这里，agent做出act动作后，数据已经被储存了，另外注意这里使用的是old_policity_act做的
            with torch.no_grad():
                action,action_prob = agent.act(state)  ### 按照策略网络输出的概率随机采样一个动作
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(action_prob)

            next_state, reward, done, _, _ = env.step(action)  ### 与环境state进行交互，输出reward 和 环境next_state
            state = next_state
            rewards.append(reward)  ### 记录每一个动作的reward
            memory.rewards.append(reward)
            memory.is_dones.append(done)

            if len(memory.rewards) >= 1200:
                agent.update(memory)

                memory.clear_memory()

            if done or len(rewards) > 1024:
                rewards_list.append(np.sum(rewards))
                #print('游戏结束')
                break
    print(f"epoch: {i} ,rewards looks like ", rewards_list[-1])

plt.plot(range(len(rewards_list)),rewards_list)
plt.show()
plt.close()
env = gym.make('LunarLander-v2', render_mode='human')
for episode in range(EPISODE_PER_BATCH):
    ### 重开一把游戏
    state = env.reset()[0]
    step = 0
    while True:
        step += 1
        # 这里，agent做出act动作后，数据已经被储存了，另外注意这里使用的是old_policity_act做的
        action,action_prob = agent.act(state)  ### 按照策略网络输出的概率随机采样一个动作
        # agent与环境进行一步交互
        state, reward, terminated, truncated, info = env.step(action)
        #print('state = {0}; reward = {1}'.format(state, reward))
        # 判断当前episode 是否完成
        if terminated or step >= 600:
            print('游戏结束')
            break
        time.sleep(0.01)

print(np.mean(rewards_list))