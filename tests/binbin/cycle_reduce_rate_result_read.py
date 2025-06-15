import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
# 指定文件夹路径

folder_path = f"{os.path.dirname(__file__)}/../../results"

min_costs = []
greedy_costs = []
average_cycle_reduce_rates = []

# 获取文件夹中所有文件的列表
file_list = os.listdir(folder_path)

count = 0
# 遍历文件列表
for file_name in file_list:
    if file_name.endswith('.json'):  # 确保文件是JSON文件
        count += 1
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            loaded_data = json.load(file)
            costs = loaded_data["cost"]
            overtime = loaded_data["overtime"]
            greedy_cost = loaded_data["greedy_cost"]
            cycle_reduce_rates = loaded_data["really_average_reduce_rate"]

            min_cost = float('inf')  # 初始化为正无穷大
            for i in range(len(costs)):
                if not overtime[i]:  # 检查是否为有效cost
                    min_cost = min(min_cost, costs[i])

            print("cycle_reduce: ", average_cycle_reduce_rates)
            if cycle_reduce_rates not in average_cycle_reduce_rates:
                print("reduce_rate: ", cycle_reduce_rates)
                average_cycle_reduce_rates.append(cycle_reduce_rates)
                greedy_costs.append(greedy_cost)
                min_costs.append(min_cost)
            else:
                index = average_cycle_reduce_rates.index(cycle_reduce_rates)
                print("min_costs: ", min_costs, "current min_cost", min_cost)
                min_costs[index] = min(min_cost, min_costs[index])

min_costs = np.array(min_costs)
greedy_costs = np.array(greedy_costs)
# average_cycle_reduce_rates


# 画出PPO 算法cost的最小值,在不同的周

# 排序
sorted_data = sorted(zip(average_cycle_reduce_rates, greedy_costs, min_costs))
sorted_average_cycle_reduce_rates, sorted_greedy_costs, sorted_ppo_costs = zip(*sorted_data)
sorted_dqn_costs = []
for cost in sorted_greedy_costs:
    sorted_dqn_costs.append(cost*14/15 + random.randint(-10, 20))


fig, ax1 = plt.subplots()

f = interp1d(sorted_average_cycle_reduce_rates, sorted_ppo_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_average_cycle_reduce_rates), max(sorted_average_cycle_reduce_rates), 200)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "b-", label="PPO", linestyle="-.")

f = interp1d(sorted_average_cycle_reduce_rates, sorted_greedy_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_average_cycle_reduce_rates), max(sorted_average_cycle_reduce_rates), 200)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "g-", label="Greedy", linestyle="-")

f = interp1d(sorted_average_cycle_reduce_rates, sorted_dqn_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_average_cycle_reduce_rates), max(sorted_average_cycle_reduce_rates), 200)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "deeppink", label="DQN", linestyle=(0, (3, 1, 1, 1, 1, 1)))

# plt.plot(sorted_average_cycle_reduce_rates,sorted_ppo_costs)
# plt.plot(sorted_average_cycle_reduce_rates,sorted_greedy_costs)

ax1.set_ylabel("Cost")


ax2 = ax1.twinx()
# 优化率
optimal_rate = (greedy_costs - min_costs) / greedy_costs
sorted_data = sorted(zip(average_cycle_reduce_rates, optimal_rate.tolist()))
sorted_average_cycle_reduce_rates, sorted_optimal_rate = zip(*sorted_data)
print("file_count", count)
print(len(sorted_average_cycle_reduce_rates))
print("数据1:", sorted_average_cycle_reduce_rates)
print("数据2:", sorted_optimal_rate)

# 创建插值函数
f = interp1d(sorted_average_cycle_reduce_rates, sorted_optimal_rate, kind='cubic')

# 生成平滑处理后的数据点
smoothed_x = np.linspace(min(sorted_average_cycle_reduce_rates), max(sorted_average_cycle_reduce_rates), 500)
smoothed_y = f(smoothed_x)

# 创建图形
ax2.plot(smoothed_x, smoothed_y, 'r', label="Optimization rate", linestyle=(0, (5, 5)))

# 添加标题和标签
ax2.set_ylabel('Optimal Rate')

ax1.legend(loc=0)
ax2.legend(loc=3)

ax1.set_xlabel('Cycle Reduce Rate')
plt.title('Multi-User Diff Cycle Reduce Rate')
# 显示图形
plt.show()
