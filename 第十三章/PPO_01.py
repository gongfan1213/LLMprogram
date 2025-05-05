"""
需要安装以下
pip install gym[box2d]
pip install gym

"""

import gym
import time

# 环境初始化
env = gym.make('LunarLander-v2', render_mode='human')

if True:
    state = env.reset()
    while True:
        # 渲染画面
        # env.render()
        # 从动作空间随机获取一个动作
        action = env.action_space.sample()
        # agent与环境进行一步交互
        observation, reward, done, _ , _= env.step(action)
        print('state = {0}; reward = {1}'.format(state, reward))
        # 判断当前episode 是否完成
        if done:
            print('游戏结束')
            break
        time.sleep(0.01)
# 环境结束
env.close()