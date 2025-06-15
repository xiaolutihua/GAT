import json
import numpy as np
import torch
import os
from binbin.algo import rl_utils
from binbin.algo.TRPO import TRPO
import binbin.algo.rl_utils
from binbin.builder import get_my_env
from binbin.env import BinbinSimEnv
from binbin.gym_env import GymEnv
from binbin.algo.alog_greedy import fastest_greedy
import uuid


def difference_learn_rate(using_gat=True, using_bc=True, save_path=None):
    num_episodes = 300
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    critic_lr_base = 1
    kl_constraint = 5e-6
    alpha = 0.5

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iot_num = 6
    fog_num = 5
    cycle_reduce_rate = 0.15

    torch.manual_seed(0)
    env = GymEnv(
        encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
        encoder_task_state=using_gat,
        encoder_task_dims=5,
        iot_num=iot_num,
        fog_num=fog_num,
        cycle_reduce_rate=cycle_reduce_rate,
        # time_unit_fix=1100  # 测试使用固定的time_unit
    )
    state_dim = 5 + (3 * env.all_dev_len)  # 状态的维度
    action_dim = env.action_space_len  # 动作的维度

    observations = []
    actions = []
    # 采集greedy的策略 , (state, action) 对
    obs, info = env.reset()
    print(f'app周期最小公倍数={env.sim.app_cycle_lcm}')
    print("=================================")
    print("采集greedy算法轨迹")
    cost = 0
    while True:
        action = fastest_greedy(info)
        observations.append(obs)
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        cost += info["cost"]
        if terminated or truncated:
            observation, info = env.reset()

            break
    greedy_overtime = terminated
    print(f"贪心算法的cost:{cost}")
    observations = np.array(observations)
    actions = np.array(actions)
    print("采集完成")

    print("=================================")
    print("克隆greedy策略")
    n_iterations = 500
    batch_size = 64

    learn_rate_count = 0
    learn_rate_count_total = 6
    while learn_rate_count < learn_rate_count_total:
        learn_rate_count += 1
        # critic_lr = critic_lr_base / 10 ** learn_rate_count
        critic_lr = 1e-2

        agent = TRPO(hidden_dim, state_dim, action_dim, lmbda,
                     kl_constraint, alpha, critic_lr, gamma, device)
        # BC 进行预训练
        # if using_bc:
        # for bc_iteration in range(n_iterations):
        #     sample_indices = np.random.randint(low=0,
        #                                        high=observations.shape[0],
        #                                        size=batch_size)
        #     agent.learn(observations[sample_indices], actions[sample_indices])

        print("行为克隆完成")

        print("=================================")
        print("开始进行强化学习训练")
        return_list, costs, overtimes = rl_utils.train_on_policy_agent_on_mec_env(env, agent, num_episodes,
                                                                                  truncate=False)

        print("总奖励", return_list)
        print("总花费", costs)
        print("超时?", overtimes)
        data = {
            "reward": return_list,
            "cost": costs,
            "overtime": overtimes,
            "really_average_reduce_rate": env.average_reduce_rate,
            "greedy_cost": cost,
            "greedy_overtime": greedy_overtime,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "algo": "trpo",
        }
        # 设置保存的名称格式如下 {iot_num}_{fog_num}_{time_unit}_{cycle_reduce_rate}_{uuid.uuid1()}
        if save_path is None:
            file_name = f'{os.path.dirname(__file__)}/../../../results/ppo/cycle/' \
                        f'C_{iot_num}_{fog_num}_{env.time_unit}_{cycle_reduce_rate}_{uuid.uuid1()}.json'
        else:
            file_name = f"{save_path}/C_{iot_num}_{fog_num}_{env.time_unit}_{cycle_reduce_rate}_{uuid.uuid1()}.json"
        print(f"实验结果保存到{file_name}，内容：{data}")
        with open(file_name, 'w') as file:
            json.dump(data, file)


if __name__ == '__main__':
    difference_learn_rate(
        using_gat=True,
        using_bc=False,
        save_path=f'{os.path.dirname(__file__)}/../../../results/TRPO/algo/learn_rate',
    )
