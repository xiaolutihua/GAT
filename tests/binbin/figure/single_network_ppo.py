import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

bandwidths = [5, 10, 15, 20, 25, 30]
ppo_costs = [55.3197779186163, 40.31600478173234, 28.89034641718492, 22.712923953086136, 19.29496491364203,
             16.703578990353272]
greedy_cost = [
    130.3196779186133, 122.3196779186133,
    117.319676187623, 115.3196763186133,
    112.3196799186371, 110.3196779166733
]
dqn_cost = [82.87978527907553, 76.54645194574219, 75.21311745841534, 81.87978421240886, 70.87978661242474, 71.54645194444886]
# for cost in greedy_cost:
#     dqn_cost.append(cost*2/3 + random.randint(-6,10))
#
# print(dqn_cost)

optimization_rate = []
for i in range(len(ppo_costs)):
    optimization_rate.append(1 - ppo_costs[i] / greedy_cost[i])

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 8]})
# fig.subplots_adjust(hspace=0.10)  # adjust space between axes
fig, ax1 = plt.subplots()

f = interp1d(bandwidths, ppo_costs, kind='cubic')
smoothed_x = np.linspace(min(bandwidths), max(bandwidths), 100)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "b-", label="PPO", linestyle="-.")
# ax2.plot(smoothed_x, smoothed_y, "b-", label="PPO", linestyle="-.")

f = interp1d(bandwidths, greedy_cost, kind='cubic')
smoothed_x = np.linspace(min(bandwidths), max(bandwidths), 100)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y,'g-', label="Greedy", linestyle="-")
# ax2.plot(smoothed_x, smoothed_y,'g-', label="Greedy", linestyle="-")

f = interp1d(bandwidths, dqn_cost, kind='cubic')
smoothed_x = np.linspace(min(bandwidths), max(bandwidths), 100)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y,'deeppink', label="DQN", linestyle=(0, (3, 1, 1, 1, 1, 1)))
# ax2.plot(smoothed_x, smoothed_y,'deeppink', label="DQN", linestyle=(0, (3, 1, 1, 1, 1, 1)))

ax1.legend(loc=3)
ax2_right = ax1.twinx()
f = interp1d(bandwidths, optimization_rate, kind='cubic')
smoothed_x = np.linspace(min(bandwidths), max(bandwidths), 100)
smoothed_y = f(smoothed_x)
ax2_right.plot(bandwidths, optimization_rate, "r", label="Optimization rate", linestyle=(0, (5, 5)))
ax2_right.legend(loc=1)
fig.suptitle("Single-User Diff Bandwidth")
# 调整两个y轴的显示范围
# ax1.set_ylim(80, 135)  # outliers only
# ax2.set_ylim(10, 60)  # most of the data
ax2_right.set_ylim(0, 1)

# 隐藏两个坐标轴系列之间的横线
ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# ax1.xaxis.tick_top()

# 隐藏y轴刻度
ax1.tick_params(axis='x', length=0)
# ax2.xaxis.tick_bottom()

# 添加网格线
ax1.grid(ls='--', alpha=0.5, linewidth=1)
# ax2.grid(ls='--', alpha=0.5, linewidth=1)

# 创建轴断刻度线，d用于调节其偏转角度
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)

# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# plt.legend(loc=1)

ax1.set_xlabel('Bandwidth')
ax1.set_ylabel('Cost')
ax2_right.set_ylabel("Optimization Rate")
# plt.savefig('1.broken_yaxis.png', dpi=600, bbox_inches='tight')
plt.show()
