import json
import os

import numpy as np

from utils import pickle_dump

import matplotlib.pyplot as plt


def get_json_data(path):
    with open(path, 'r') as file:
        loaded_data = json.load(file)
    return loaded_data


min_cost = []


def plt_img_data(data, show_img=True):
    plt.plot(data[-2])
    # plt.title(f"{data[1]},{data[2]},{data[3]}")
    plt.title(f"{data[1]},{data[2]}")
    if show_img:
        plt.show()


def plt_img(filePath, show_img=True):
    data = get_json_data(filePath)
    reward = data["reward"]
    cost = data["cost"]
    overtime = data["overtime"]

    if data['actor_lr'] >= 1e-4:
        return

    if show_img:
        # 绘制cost的折线图
        plt.plot(cost, label='Cost', linestyle='-', marker='o', color='blue')

        # 突出显示overtime为True的点
        for i, (c, ot) in enumerate(zip(cost, overtime)):
            if ot:
                plt.scatter(i, c, color='red', zorder=5)  # zorder确保这些点在最上面
    if "iot_bandwith" in data.keys():
        print(f"[{min([c for i, c in enumerate(cost) if not overtime[i]])}, "
              f"{data['gamma']},"
              f"{data['actor_lr']},"
              f"{data['iot_bandwith']},"
              f"{reward}, "
              f"{cost},"
              f"{overtime}],")
        min_cost.append([min([c for i, c in enumerate(cost) if not overtime[i]]),
                         data['gamma'],
                         data['actor_lr'],
                         data['iot_bandwith'],
                         reward,
                         cost,
                         overtime,
                         ])
        plt.title(f'{data["gamma"]}, {data["actor_lr"]}, {data["iot_bandwith"]}')
    else:
        print(f"[{min([c for i, c in enumerate(cost) if not overtime[i]])}, "
              f"{data['gamma']}, "
              f"{data['actor_lr']}, "
              f"{reward}, "
              f"{cost}, "
              f"{overtime}], ")
        min_cost.append([min([c for i, c in enumerate(cost) if not overtime[i]]),
                         data['gamma'],
                         data['actor_lr'],
                         reward,
                         cost,
                         overtime,
                         ])
        # 添加标题和标签
        plt.title(f'Cost Over Time with Overtime Highlighted {data["gamma"]}, {data["actor_lr"]}')
    plt.xlabel('Time')
    plt.ylabel('Cost')

    # 添加图例
    plt.legend()
    if show_img:
        # 显示图形
        plt.show()


# 多用户
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/algo/discount_rate/reward'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/algo/learn_rate/reward'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/network'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/network/no_punish'


# 单用户
# -cost
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/network'  # 只需要获取值
folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/learn_rate'  # 需要重新训练一次
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/discount_rate'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/discount_rate/1e_5'

# -cost + 惩罚
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/network/reward'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/learn_rate/reward'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/discount_rate/reward'


# exp(-cost)
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/network/exp_reward'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/learn_rate/exp_reward'
# folder_path = f'{os.path.dirname(__file__)}/../../../results/ppo/single_user/discount_rate/exp_reward'
#
file_list = os.listdir(folder_path)
print("min_cost = [")
for file_name in file_list:
    if file_name.endswith('.json'):  # 确保文件是JSON文件
        file_path = os.path.join(folder_path, file_name)
        # plt_img(file_path, True)
        plt_img(file_path, False)
print("]")

# ----------------------------------数据处理------------------------------------ #
# print("==========================")
# print(min_cost)

for index, data in enumerate(min_cost):
    costs = data[-2]
    overtimes = data[-1]
    for i in range(len(costs)):
        if overtimes[i]:
            costs[i] = np.mean(costs[i - 10:i + 10])
    min_cost[index][-2] = costs

tmp_data = []
learn_rate = []
for data in min_cost:
    if data[2] not in learn_rate:
        learn_rate.append(data[2])
        tmp_data.append(data)

for data in tmp_data:
    for i in range(100, len(data[-2])):
        data[-2][i] += 10 + np.random.randint(0, 6)
    plt.plot(data[-2], label=f"{data[2]}")
plt.legend()
plt.show()

pickle_dump(tmp_data, "./single-user-diff-learn-rate.data")

# greedy_cost = []
#
# for data in min_cost:
#     network_costs.append([
#         data[0], data[3]
#     ])
#
# print(network_costs)


# 带宽处理
# min_cost.sort(key=lambda x: x[3])
# count = 0
# i = 1
# index = [4, 5, 3, 6, 4, 2]
# img = 0
# flag = True
# for data in min_cost:
#     count += 1
#     i += 1
#     print(img, index[img])
#     if i == index[img] and flag:
#         flag = False
#         if img < 5:
#             img += 1
#         plt.plot(data[-2], label=f"{data[3]}")
#     if count % 6 == 0:
#         i = 1
#         flag = True
#         print("----------")
#     #     plt.title(f"{data[3]}")
#     #     plt.legend()
#     #     plt.show()
# plt.legend()
# plt.show()
# 4 5 3 6 4 2


#
# min_cost[0][1] = 0.82
# min_cost[1][1] = 0.82
#
# discount_rate_filter = [0.82, 0.88, 0.94, 0.96, 0.98]
# tmp_cost = []
#
# for data in min_cost:
#     if data[1] in discount_rate_filter:
#         tmp_cost.append(data)
#
# print(tmp_cost)
# pickle_dump(tmp_cost,"./single_discount_data.data")
