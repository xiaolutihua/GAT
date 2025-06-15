import os
import json
import uuid

import numpy as np
import torch

from binbin.algo import rl_utils
from binbin.algo.PPO import PPO
import binbin.algo.rl_utils
from binbin.builder import get_my_env
from binbin.env import BinbinSimEnv
from binbin.gym_env import GymEnv
from binbin.algo.alog_greedy import fastest_greedy


def difference_cycle_reduce(using_gat=True, using_bc=True, save_path=None):
    actor_lr = 1e-5
    critic_lr = 2e-5
    # rate_episodes 定义的是每个周期缩短率进行多少次训练， num_episodes限定每次限定进行多少轮训练
    num_episodes = 500
    rate_episodes = 6
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iot_num = 6
    fog_num = 5
    cycle_reduce_rate = 0.15

    torch.manual_seed(0)
    rate_count = 0
    flag_greedy_run = True
    flag_greedy_error = False
    greedy_cost = 0
    greedy_overtime = False

    observations = []
    actions = []
    while cycle_reduce_rate < 1:
        if rate_count >= rate_episodes:
            rate_count = 0
            flag_greedy_error = False
            flag_greedy_run = True
            cycle_reduce_rate += 0.05
        rate_count += 1
        if using_gat:
            env = GymEnv(
                encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
                encoder_task_state=True,
                encoder_task_dims=5,
                iot_num=iot_num,
                fog_num=fog_num,
                cycle_reduce_rate=cycle_reduce_rate,
                # time_unit_fix=1100  # 测试使用固定的time_unit
            )
        else:
            env = GymEnv(
                encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
                encoder_task_state=False,
                encoder_task_dims=5,
                iot_num=iot_num,
                fog_num=fog_num,
                cycle_reduce_rate=cycle_reduce_rate,
                # time_unit_fix=1100  # 测试使用固定的time_unit
            )
        state_dim = 5 + (3 * env.all_dev_len)  # 状态的维度
        action_dim = env.action_space_len  # 动作的维度
        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                    epochs, eps, gamma, device)

        # 采集greedy的策略 , (state, action) 对
        obs, info = env.reset()
        print(f'app周期最小公倍数={env.sim.app_cycle_lcm}')
        if flag_greedy_run:
            print(f"current cycle_reduce_rate:{cycle_reduce_rate} and iot_num:{iot_num}")
            flag_greedy_run = False
            observations = []
            actions = []
            print("=================================")
            print("采集greedy算法轨迹")
            greedy_cost = 0
            total_step = 0
            while True:
                action = fastest_greedy(info)
                observations.append(obs)
                actions.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                total_step += 1
                if total_step > env.total_app_nums * 20:
                    flag_greedy_error = True
                    print("程序异常")
                    # break
                greedy_cost += info["cost"]
                if terminated or truncated:
                    observation, info = env.reset()
                    break
            greedy_overtime = terminated
            print(f"贪心算法的cost:{greedy_cost} 是否超时:{greedy_overtime}")
            print("采集完成")
        if flag_greedy_error:
            print("程序错误暂时跳过")
            continue
        observations = np.array(observations)
        actions = np.array(actions)
        print("=================================")
        print("克隆greedy策略")
        n_iterations = 500
        batch_size = 64
        # BC 进行预训练
        if using_bc:
            for bc_iteration in range(n_iterations):
                sample_indices = np.random.randint(low=0,
                                                   high=observations.shape[0],
                                                   size=batch_size)
                agent.learn(observations[sample_indices], actions[sample_indices])
            print("行为克隆完成")

        print("=================================")
        print("开始进行强化学习训练")
        return_list, costs, overtimes = rl_utils.train_on_policy_agent_on_mec_env(env, agent, num_episodes)

        data = {
            "reward": return_list,
            "cost": costs,
            "overtime": overtimes,
            "really_average_reduce_rate": env.average_reduce_rate,
            "greedy_cost": greedy_cost,
            "greedy_overtime": greedy_overtime,
            "algo": "ppo",
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


def difference_learn_rate(using_gat=True, using_bc=True, save_path=None):
    # actor_lr_base = 1  # 0.1 0.01 0.001 0.0001 0.00001
    # critic_lr_base = 2  # 0.2 0.02 0.002 0.0002 0.00002
    actor_lr_base = 1e-5  # 0.1 0.01 0.001 0.0001 0.00001
    critic_lr_base = 2e-5  # 0.2 0.02 0.002 0.0002 0.00002
    # rate_episodes 定义的是每个周期缩短率进行多少次训练， num_episodes限定每次限定进行多少轮训练
    num_episodes = 600
    rate_episodes = 1
    hidden_dim = 128
    gamma = 0.94
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cycle_reduce_rate = 0.15

    iot_num = 1
    fog_num = 5
    learn_rate_count = 0
    learn_rate_count_total = 6

    torch.manual_seed(0)
    flag_greedy_error = False
    while learn_rate_count < learn_rate_count_total:
        learn_rate_count += 1
        # actor_lr = actor_lr_base / 10 ** (learn_rate_count_total - learn_rate_count + 1)
        # critic_lr = critic_lr_base / 10 ** (learn_rate_count_total - learn_rate_count + 1)
        actor_lr = actor_lr_base * learn_rate_count
        critic_lr = critic_lr_base * learn_rate_count
        print(f"==================current learn rate : {actor_lr} {critic_lr} ==============================")
        env = GymEnv(
            encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
            encoder_task_state=using_gat,
            encoder_task_dims=5,
            iot_num=iot_num,
            fog_num=fog_num,
            cycle_reduce_rate=cycle_reduce_rate,
            # time_unit_fix=1100  # 测试使用固定的time_unit
            task_nums=20 if iot_num > 1 else 50
        )
        state_dim = 5 + (3 * env.all_dev_len)  # 状态的维度
        action_dim = env.action_space_len  # 动作的维度

        # 采集greedy的策略 , (state, action) 对
        obs, info = env.reset()
        print(f'app周期最小公倍数={env.sim.app_cycle_lcm}')
        print(f"current cycle_reduce_rate:{cycle_reduce_rate} and iot_num:{iot_num} and actor_lr:{actor_lr}, "
              f"critic_lr:{critic_lr}, gamma: {gamma}")
        observations = []
        actions = []
        print("=================================")
        print("采集greedy算法轨迹")
        greedy_cost = 0
        total_step = 0
        while True:
            action = fastest_greedy(info)
            observations.append(obs)
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_step += 1
            if total_step > env.total_app_nums * 20:
                flag_greedy_error = True
                print("程序异常")
                # break
            greedy_cost += info["cost"]
            if terminated or truncated:
                observation, info = env.reset()
                break
        greedy_overtime = terminated
        print(f"贪心算法的cost:{greedy_cost} 是否超时:{greedy_overtime}")
        print("采集完成")
        if flag_greedy_error:
            print("程序错误暂时跳过")
            continue
        observations = np.array(observations)
        actions = np.array(actions)

        n_iterations = 500
        batch_size = 64
        # BC 进行预训练
        rate_count = 0
        while rate_count < rate_episodes:
            rate_count += 1
            agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                        epochs, eps, gamma, device)
            print("=================================")
            print("克隆greedy策略")
            if using_bc:
                for bc_iteration in range(n_iterations):
                    sample_indices = np.random.randint(low=0,
                                                       high=observations.shape[0],
                                                       size=batch_size)
                    agent.learn(observations[sample_indices], actions[sample_indices])
                print("行为克隆完成")

            print("=================================")
            print("开始进行强化学习训练")
            return_list, costs, overtimes = rl_utils.train_on_policy_agent_on_mec_env(env, agent, num_episodes,
                                                                                      truncate=False)
            data = {
                "reward": return_list,
                "cost": costs,
                "overtime": overtimes,
                "really_average_reduce_rate": env.average_reduce_rate,
                "greedy_cost": greedy_cost,
                "greedy_overtime": greedy_overtime,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "gamma": gamma,
                "edge_only": env.only_mec_cost(),
                "algo": "ppo",
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


def difference_discount_rate(using_gat=True, using_bc=True, save_path=None):
    actor_lr = 1e-5  # 0.1 0.01 0.001 0.0001 0.00001
    critic_lr = 2e-5  # 0.2 0.02 0.002 0.0002 0.00002
    # rate_episodes 定义的是每个周期缩短率进行多少次训练， num_episodes限定每次限定进行多少轮训练
    num_episodes = 600
    rate_episodes = 2
    hidden_dim = 128
    gamma_base = 1
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cycle_reduce_rate = 0.15

    iot_num = 1
    fog_num = 5
    learn_rate_count = 0
    learn_rate_count_total = 10

    torch.manual_seed(0)
    flag_greedy_error = False
    while learn_rate_count < learn_rate_count_total:
        gamma = gamma_base - 0.02 * learn_rate_count
        learn_rate_count += 1
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

        # 采集greedy的策略 , (state, action) 对
        obs, info = env.reset()
        print(f'app周期最小公倍数={env.sim.app_cycle_lcm}')
        print(f"current cycle_reduce_rate:{cycle_reduce_rate} and iot_num:{iot_num} and actor_lr:{actor_lr}, "
              f"critic_lr:{critic_lr}, gamma: {gamma}")
        observations = []
        actions = []
        print("=================================")
        print("采集greedy算法轨迹")
        greedy_cost = 0
        total_step = 0
        while True:
            action = fastest_greedy(info)
            observations.append(obs)
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_step += 1
            if total_step > env.total_app_nums * 20:
                flag_greedy_error = True
                print("程序异常")
                # break
            greedy_cost += info["cost"]
            if terminated or truncated:
                observation, info = env.reset()
                break
        greedy_overtime = terminated
        print(f"贪心算法的cost:{greedy_cost} 是否超时:{greedy_overtime}")
        print("采集完成")
        if flag_greedy_error:
            print("程序错误暂时跳过")
            continue
        observations = np.array(observations)
        actions = np.array(actions)

        n_iterations = 500
        batch_size = 64
        # BC 进行预训练
        rate_count = 0
        while rate_count < rate_episodes:
            print(rate_count, gamma, actor_lr)
            rate_count += 1

            agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                        epochs, eps, gamma, device)
            print("=================================")
            print("克隆greedy策略")
            if using_bc:
                for bc_iteration in range(n_iterations):
                    sample_indices = np.random.randint(low=0,
                                                       high=observations.shape[0],
                                                       size=batch_size)
                    agent.learn(observations[sample_indices], actions[sample_indices])
                print("行为克隆完成")

            print("=================================")
            print("开始进行强化学习训练")
            return_list, costs, overtimes = rl_utils.train_on_policy_agent_on_mec_env(env, agent, num_episodes,
                                                                                      truncate=False)
            data = {
                "reward": return_list,
                "cost": costs,
                "overtime": overtimes,
                "really_average_reduce_rate": env.average_reduce_rate,
                "greedy_cost": greedy_cost,
                "greedy_overtime": greedy_overtime,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "gamma": gamma,
                "edge_only": env.only_mec_cost(),
                "algo": "ppo",
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


def difference_iot_devices(using_gat=True, using_bc=True, save_path=None):
    actor_lr = 1e-5
    critic_lr = 2e-5
    # devices_episodes 定义的是每个终端设备数量进行多少次训练， num_episodes限定每次限定进行多少轮训练
    num_episodes = 500
    devices_episodes = 10
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iot_num = 9
    iot_num_max = 16
    fog_num = 5
    cycle_reduce_rate = 0.6

    torch.manual_seed(0)
    device_count = 0
    flag_greedy_run = True
    flag_greedy_error = False
    greedy_cost = 0
    greedy_overtime = False

    observations = []
    actions = []
    while iot_num <= iot_num_max:
        if device_count >= devices_episodes:
            device_count = 0
            flag_greedy_error = False
            flag_greedy_run = True
            iot_num += 1
        device_count += 1
        if using_gat:
            env = GymEnv(
                encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
                encoder_task_state=True,
                encoder_task_dims=5,
                iot_num=iot_num,
                fog_num=fog_num,
                cycle_reduce_rate=cycle_reduce_rate,
                # time_unit_fix=1100  # 测试使用固定的time_unit
            )
        else:
            env = GymEnv(
                encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
                encoder_task_state=False,
                encoder_task_dims=5,
                iot_num=iot_num,
                fog_num=fog_num,
                cycle_reduce_rate=cycle_reduce_rate,
                # time_unit_fix=1100  # 测试使用固定的time_unit
            )
        state_dim = 5 + (3 * env.all_dev_len)  # 状态的维度
        action_dim = env.action_space_len  # 动作的维度
        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                    epochs, eps, gamma, device)

        # 采集greedy的策略 , (state, action) 对
        obs, info = env.reset()
        print(f'app周期最小公倍数={env.sim.app_cycle_lcm}')
        if flag_greedy_run:
            print(f"current cycle_reduce_rate:{cycle_reduce_rate} and iot_num:{iot_num}")
            flag_greedy_run = False
            observations = []
            actions = []
            print("=================================")
            print("采集greedy算法轨迹")
            greedy_cost = 0
            total_step = 0
            while True:
                action = fastest_greedy(info)
                observations.append(obs)
                actions.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                total_step += 1
                if total_step > env.total_app_nums * 20:
                    flag_greedy_error = True
                    print("程序异常")
                    # break
                greedy_cost += info["cost"]
                if terminated or truncated:
                    observation, info = env.reset()
                    break
            greedy_overtime = terminated
            print(f"贪心算法的cost:{greedy_cost} 是否超时:{greedy_overtime}")
            print("采集完成")
        if flag_greedy_error:
            print("程序错误暂时跳过")
            continue
        observations = np.array(observations)
        actions = np.array(actions)
        print("=================================")
        print("克隆greedy策略")
        n_iterations = 500
        batch_size = 64
        # BC 进行预训练
        if using_bc:
            for bc_iteration in range(n_iterations):
                sample_indices = np.random.randint(low=0,
                                                   high=observations.shape[0],
                                                   size=batch_size)
                agent.learn(observations[sample_indices], actions[sample_indices])
            print("行为克隆完成")

        print("=================================")
        print("开始进行强化学习训练")
        return_list, costs, overtimes = rl_utils.train_on_policy_agent_on_mec_env(env, agent, num_episodes)

        data = {
            "reward": return_list,
            "cost": costs,
            "overtime": overtimes,
            "really_average_reduce_rate": env.average_reduce_rate,
            "greedy_cost": greedy_cost,
            "greedy_overtime": greedy_overtime,
            "algo": "ppo",
        }
        # 设置保存的名称格式如下 {iot_num}_{fog_num}_{time_unit}_{cycle_reduce_rate}_{uuid.uuid1()}
        if save_path is None:
            file_name = f'{os.path.dirname(__file__)}/../../../results/ppo/device/' \
                        f'D_{iot_num}_{fog_num}_{env.time_unit}_{cycle_reduce_rate}_{uuid.uuid1()}.json'
        else:
            file_name = f"{save_path}/D_{iot_num}_{fog_num}_{env.time_unit}_{cycle_reduce_rate}_{uuid.uuid1()}.json"
        print(f"实验结果保存到{file_name}，内容：{data}")
        with open(file_name, 'w') as file:
            json.dump(data, file)


def difference_iot_bandwidth(using_gat=True, using_bc=True, save_path=None):
    actor_lr = 1e-3
    critic_lr = 2e-3
    # rate_episodes 定义的是每个周期缩短率进行多少次训练， num_episodes限定每次限定进行多少轮训练
    num_episodes = 600
    rate_episodes = 6
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cycle_reduce_rate = 0.15

    iot_num = 1
    fog_num = 5
    learn_rate_count = 0
    learn_rate_count_total = 6

    torch.manual_seed(0)
    flag_greedy_error = False
    while learn_rate_count < learn_rate_count_total:
        learn_rate_count += 1
        env = GymEnv(
            encoder_weight=f"{os.path.dirname(__file__)}/../../../tests/gat/encoder-checkpoint10.pt",
            encoder_task_state=using_gat,
            encoder_task_dims=5,
            iot_num=iot_num,
            fog_num=fog_num,
            cycle_reduce_rate=cycle_reduce_rate,
            # time_unit_fix=1100  # 测试使用固定的time_unit
            iot_bandwidth=learn_rate_count * 5
        )
        state_dim = 5 + (3 * env.all_dev_len)  # 状态的维度
        action_dim = env.action_space_len  # 动作的维度

        # 采集greedy的策略 , (state, action) 对
        obs, info = env.reset()
        print(f'app周期最小公倍数={env.sim.app_cycle_lcm}')
        print(f"current cycle_reduce_rate:{cycle_reduce_rate} and iot_num:{iot_num} and actor_lr:{actor_lr}, "
              f"critic_lr:{critic_lr}, gamma: {gamma}")
        observations = []
        actions = []
        print("=================================")
        print("采集greedy算法轨迹")
        greedy_cost = 0
        total_step = 0
        while True:
            action = fastest_greedy(info)
            observations.append(obs)
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_step += 1
            if total_step > env.total_app_nums * 20:
                flag_greedy_error = True
                print("程序异常")
                # break
            greedy_cost += info["cost"]
            if terminated or truncated:
                observation, info = env.reset()
                break
        greedy_overtime = terminated
        print(f"贪心算法的cost:{greedy_cost} 是否超时:{greedy_overtime}")
        print("采集完成")
        if flag_greedy_error:
            print("程序错误暂时跳过")
            continue
        observations = np.array(observations)
        actions = np.array(actions)

        n_iterations = 500
        batch_size = 64
        # BC 进行预训练
        rate_count = 0
        while rate_count < rate_episodes:
            rate_count += 1
            agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                        epochs, eps, gamma, device)
            print("=================================")
            print("克隆greedy策略")
            if using_bc:
                for bc_iteration in range(n_iterations):
                    sample_indices = np.random.randint(low=0,
                                                       high=observations.shape[0],
                                                       size=batch_size)
                    agent.learn(observations[sample_indices], actions[sample_indices])
                print("行为克隆完成")

            print("=================================")
            print("开始进行强化学习训练")
            return_list, costs, overtimes = rl_utils.train_on_policy_agent_on_mec_env(env, agent, num_episodes,
                                                                                      truncate=False)
            data = {
                "reward": return_list,
                "cost": costs,
                "overtime": overtimes,
                "really_average_reduce_rate": env.average_reduce_rate,
                "greedy_cost": greedy_cost,
                "greedy_overtime": greedy_overtime,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "gamma": gamma,
                "edge_only": env.only_mec_cost(),
                "iot_bandwith": learn_rate_count * 5,
                "algo": "ppo",
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
    # difference_cycle_reduce(
    #     using_gat=True,
    #     using_bc=True,
    #     save_path=f"{os.path.dirname(__file__)}/../../../results/ppo/cycle/without_bc",  # 禁止使用绝对路径
    # )
    # difference_iot_devices(
    #     using_gat=True,
    #     using_bc=True,
    #     save_path=f"{os.path.dirname(__file__)}/../../../results/ppo/device", # 禁止使用绝对路径
    # )
    difference_learn_rate(
        using_gat=False,
        using_bc=True,
        save_path=f"{os.path.dirname(__file__)}/../../../results/ppo/single_user/learn_rate",  # 禁止使用绝对路径
    )
    # difference_discount_rate(
    #     using_gat=False,
    #     using_bc=True,
    #     save_path=f"{os.path.dirname(__file__)}/../../../results/ppo/single_user/discount_rate/1e_5",  # 禁止使用绝对路径
    # )
    # difference_iot_bandwidth(
    #     using_gat=False,
    #     using_bc=True,
    #     save_path=f"{os.path.dirname(__file__)}/../../../results/ppo/single_user/network/exp_reward",  # 禁止使用绝对路径
    # )
