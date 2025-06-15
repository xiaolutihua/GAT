import os
import json
import numpy as np

min_costs = []
greedy_costs = []
average_cycle_reduce_rates = []
folder_path = f'{os.path.dirname(__file__)}/../../results/ppo/cycle/without_bc'

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

print(f"min_cost : {min_costs.tolist()}\n greedy_costs: {greedy_costs.tolist()}")


optimal_rate = (greedy_costs - min_costs) / greedy_costs
sorted_data = sorted(zip(average_cycle_reduce_rates, optimal_rate.tolist(), min_costs.tolist()))
sorted_average_cycle_reduce_rates, sorted_optimal_rate, sorted_min_costs = zip(*sorted_data)
print("file_count", count)
print(len(sorted_average_cycle_reduce_rates))
print("数据1:", sorted_average_cycle_reduce_rates)
print("数据2:", sorted_optimal_rate)
print("数据3:", sorted_min_costs)

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



plt.plot(sorted_average_cycle_reduce_rates, sorted_optimal_rate, linestyle="-")
plt.show()

# 创建插值函数
f = interp1d(sorted_average_cycle_reduce_rates, sorted_optimal_rate, kind='cubic')

# 生成平滑处理后的数据点
smoothed_x = np.linspace(min(sorted_average_cycle_reduce_rates), max(sorted_average_cycle_reduce_rates), 500)
smoothed_y = f(smoothed_x)

# 创建图形
plt.plot(smoothed_x, smoothed_y, linestyle='-')

# 添加标题和标签
plt.title('Optimal Rate vs Cycle Reduce Rates (Smoothed)')
plt.xlabel('Cycle Reduce Rates')
plt.ylabel('Optimal Rate')

# 显示图形
plt.show()
