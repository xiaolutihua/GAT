"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
https://gitee.com/MorvanZhou/pytorch-A3C/blob/master/utils.py#
"""

import torch
import torch.nn as nn
from binbin.algo.a3c_utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gymnasium as gym
import os
import numpy as np
from binbin.gym_env import GymEnv
from binbin.algo.alog_greedy import fastest_greedy

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

env = gym.make('CartPole-v0')
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.n

N_S = 2
N_A = 4
N_ITERATIONS = 500


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        # policy 网络
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        # Critic 网络
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()  # 进入推理模式, 参数不升级
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()  # 进入训练模式
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def learn(self, states, actions, actor_lr=5e-4):
        # 进行行为克隆
        actor_opt = torch.optim.Adam(self.parameters(),
                                     lr=actor_lr)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions,dtype=torch.int64).view(-1, 1)
        # self.actor(states).gather(1, actions)
        logits, _ = self.forward(states)
        log_probs = torch.log(F.softmax(logits, dim=1).gather(1, actions))
        bc_loss = torch.mean(-log_probs)  # 最大似然估计
        actor_opt.zero_grad()
        bc_loss.backward()
        actor_opt.step()


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, env):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(env.STATE_DIM, env.ACTION_DIM)  # local network
        self.lnet.load_state_dict(gnet.state_dict())  # 初始化本地策略与全局策略一致
        self.env = env

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s, info = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                # if self.name == 'w00':
                #     self.env.render()
                # a = self.lnet.choose_action(v_wrap(s[None, :])) #
                a = self.lnet.choose_action(torch.tensor([s], dtype=torch.float))
                s_, r, done, truncate, info = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == '__main__':
    env = GymEnv(encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
                 encoder_task_state=True,
                 encoder_task_dims=5
                 )
    observation, info = env.reset()

    gnet = Net(env.STATE_DIM, env.ACTION_DIM)
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    observations = []
    actions = []
    print("========开始贪心算法采集轨迹===============")
    cost = 0
    # while True:
    #     action = fastest_greedy(info)
    #     observations.append(observation)
    #     actions.append(action)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     cost += info["cost"]
    #     if terminated or truncated:
    #         observation, info = env.reset()
    #         break
    # print(f"贪心算法的cost:{cost} 是否超时:{terminated}")
    # observations = np.array(observations)
    # actions = np.array(actions)
    print("采集完成")

    print("========开始行为克隆===============")
    batch_size = 64
    states = np.array(observations)
    actions = np.array(actions)
    # for bc_iteration in range(N_ITERATIONS):
    #     sample_indices = np.random.randint(low=0,
    #                                        high=states.shape[0],
    #                                        size=batch_size)
    #     gnet.learn(states[sample_indices], actions[sample_indices])

    print("================开始训练================")
    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i,
                      GymEnv(encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
                             encoder_task_state=True,
                             encoder_task_dims=5
                             )
                      )
               # for i in range(mp.cpu_count() - 1)
               for i in range(2)
               ]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

