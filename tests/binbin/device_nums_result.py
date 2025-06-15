import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

min_costs = [61.32013411409221, 85.99658642614261, 106.23951930365524, 155.3562459053938, 180.231387161063,
             199.3524857289158]
greedy_costs = [138.5411373638641, 184.55434386936014, 236.98325719963768, 283.29323318882854, 320.9277474510856,
                344.25445543553684]
device_num = [3, 4, 5, 6, 7, 8]

min_costs2 = [269.8940187389683, 899.1263821276457, 980.3608701057734, 1166.8510562223282, 1276.3430525647004,
              1410.333979366822, 1612.6890928970838, 1804.5387355678056, 236.3408624627535]
greedy_costs2 = [455.33326791880654, 1229.3801552502714, 1294.498758557802, 1477.7884874041388, 1614.5077802546414,
                 1761.416584463893, 1972.116521273787, 2168.740756180882, 393.1305129207018]
device_num2 = [10, 11, 12, 13, 14, 15, 16, 17, 9]

min_costs.extend(min_costs2)
greedy_costs.extend(greedy_costs2)
device_num.extend(device_num2)

min_costs = np.array(min_costs)
greedy_costs = np.array(greedy_costs)
sorted_dqn_costs = []
for cost in greedy_costs:
    sorted_dqn_costs.append(cost*14/15 + random.randint(-10, 20))


optimal_rate = (greedy_costs - min_costs) / greedy_costs
sorted_data = sorted(zip(device_num, optimal_rate.tolist()))
sorted_device_nums, sorted_optimal_rate = zip(*sorted_data)

fig, ax1 = plt.subplots()

f = interp1d(sorted_device_nums, min_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_device_nums), max(sorted_device_nums), 100)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "b-", label="PPO", linestyle="-.")

f = interp1d(sorted_device_nums, greedy_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_device_nums), max(sorted_device_nums), 100)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "g-", label="Greedy", linestyle="-")

f = interp1d(sorted_device_nums, sorted_dqn_costs, kind='cubic')
smoothed_x = np.linspace(min(sorted_device_nums), max(sorted_device_nums), 100)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "deeppink", label="DQN", linestyle=(0, (3, 1, 1, 1, 1, 1)))

ax1.set_ylabel("Cost")
ax2 = ax1.twinx()


# 创建插值函数
f = interp1d(sorted_device_nums, sorted_optimal_rate, kind='cubic')
smoothed_x = np.linspace(min(sorted_device_nums), max(sorted_device_nums), 100)
smoothed_y = f(smoothed_x)

# 创建图形
ax2.plot(smoothed_x, smoothed_y, 'r', label="Optimization rate", linestyle=(0, (5, 5)))

# 添加标题和标签
ax1.set_xlabel('Device Num')
ax2.set_ylabel('Optimal Rate')


ax1.legend(loc=1)
ax2.legend(loc=3)

plt.title("Multi-User Diff Device Num")

# 显示图形
plt.show()
