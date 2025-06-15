import json
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 指定文件夹路径
# folder_path = "D:\\BaiduSyncdisk\\DAG-MEC\\experiment\\results\\ppo"
folder_path = f"{os.path.dirname(__file__)}/../../../experiment/results/ppo/device"
# folder_path = "C:\\Users\\sct\\Desktop\\论文\\dag-mec\\experiment\\results\\ppo"
# folder_path = "C:\\Users\\sct\\Desktop\\论文\\dag-mec\\experiment\\results\\ppo\\device"

min_costs = []
greedy_costs = []
device_nums = []

file_list = os.listdir(folder_path)

count = 0
# 遍历文件列表
for file_name in file_list:
    if file_name.endswith('.json'):  # 确保文件是JSON文件
        count += 1
        match = re.search(r'D_(\d+)_', file_name)
        # match = re.search(r'D_(\d+)_', file_name)
        if match:
            device_number = int(match.group(1))
            print(device_number)
        else:
            print("No device number found in the file name.")
            break
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            loaded_data = json.load(file)
            costs = loaded_data["cost"]
            overtime = loaded_data["overtime"]
            greedy_cost = loaded_data["greedy_cost"]

            min_cost = float('inf')  # 初始化为正无穷大
            for i in range(len(costs)):
                if not overtime[i]:  # 检查是否为有效cost
                    min_cost = min(min_cost, costs[i])
            if device_number not in device_nums:
                device_nums.append(device_number)
                greedy_costs.append(greedy_cost)
                min_costs.append(min_cost)
            else:
                index = device_nums.index(device_number)
                print("min_costs: ", min_costs, "current min_cost", min_cost)
                min_costs[index] = min(min_cost, min_costs[index])

print("打印", min_costs, greedy_costs, device_nums)

min_costs = np.array(min_costs)
greedy_costs = np.array(greedy_costs)

# 费用
sorted_data = sorted(zip(device_nums, min_costs, greedy_costs))
sorted_device_nums, sorted_min_costs, sorted_greedy_costs = zip(*sorted_data)
f = interp1d(sorted_device_nums, sorted_min_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_device_nums), max(sorted_device_nums), 200)
smoothed_y = f(smoothed_x)
plt.plot(smoothed_x, smoothed_y, label="PPO")

f = interp1d(sorted_device_nums, sorted_greedy_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_device_nums), max(sorted_device_nums), 200)
smoothed_y = f(smoothed_x)
plt.plot(smoothed_x, smoothed_y, label="Greedy")
plt.xlabel("Device num")
plt.ylabel("Cost")
plt.title("Multi-User Diff Device Num")
plt.legend(loc=1)
plt.show()

# 优化率图片

optimal_rate = (greedy_costs - min_costs) / greedy_costs
sorted_data = sorted(zip(device_nums, optimal_rate.tolist()))
sorted_device_nums, sorted_optimal_rate = zip(*sorted_data)
print(count)

# 创建插值函数
f = interp1d(sorted_device_nums, sorted_optimal_rate, kind='cubic')

# 生成平滑处理后的数据点
smoothed_x = np.linspace(min(sorted_device_nums), max(sorted_device_nums), 500)
smoothed_y = f(smoothed_x)

# 创建图形
plt.plot(smoothed_x, smoothed_y, linestyle='-')

# 添加标题和标签
plt.title('Optimal Rate vs device_nums (Smoothed)')
plt.xlabel('Device Num')
plt.ylabel('Optimal Rate')

# 显示图形
plt.show()
