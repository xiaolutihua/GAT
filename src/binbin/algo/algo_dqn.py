import os
from random import random

import numpy as np
import torch

from binbin.gym_env import GymEnv
from dqn import ReplayBuffer, DQN
from tqdm import tqdm

lr = 1e-5
num_episodes = 600
hidden_dim = 128
gamma = 0.98
epsilon = 0.05
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

iot_num = 6
fog_num = 5
cycle_reduce_rate = 0.15

env = GymEnv(
    encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
    encoder_task_state=False,
    encoder_task_dims=5,
    iot_num=iot_num,
    fog_num=fog_num,
    cycle_reduce_rate=cycle_reduce_rate,
    # time_unit_fix=1100  # 测试使用固定的time_unit
)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = 5 + (3 * env.all_dev_len)  # 状态的维度
action_dim = env.action_space_len  # 动作的维度
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
overtimes = []
for i in range(1):
    with tqdm(total=int(num_episodes), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes)):
            episode_return = 0
            epsiode_cost = 0
            state, info = env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                epsiode_cost += info["cost"]
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(epsiode_cost)
            overtimes.append(done)
            pbar.set_postfix({
                'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                'return': '%.3f' % episode_return,
                'cost':'%.3f' % epsiode_cost,
                'overtime': done
            })
            pbar.update(1)

print(return_list)
print(overtimes)
