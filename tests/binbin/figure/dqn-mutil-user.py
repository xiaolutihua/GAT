import os
import matplotlib.pyplot as plt
from utils import pickle_load

ppo_cost = pickle_load(f"{os.path.dirname(__file__)}/../../../data/multi-user/method/ppo_cost.data")
dqn_costs = pickle_load(f"{os.path.dirname(__file__)}/../../../data/multi-user/method/dqn_costs.data")
greedy_cost = [1492.908089535172 for _ in range(len(dqn_costs))]
edge_only = [19826.23939977911 for _ in range(len(dqn_costs))]
cloud_only = [19826.23939977911 * 1.2 for _ in range(len(dqn_costs))]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 8]})
fig.subplots_adjust(hspace=0.10)  # adjust space between axe
ax1.plot(ppo_cost, label="PPO", linestyle="-", linewidth=2)
ax2.plot(ppo_cost, label="PPO", linestyle="-", linewidth=2)
ax1.plot(dqn_costs, label="DQN", linestyle="-.", linewidth=2)
ax2.plot(dqn_costs, label="DQN", linestyle="-.", linewidth=2)
ax1.plot(greedy_cost, label="Greedy", linestyle=":", linewidth=2)
ax2.plot(greedy_cost, label="Greedy", linestyle=":", linewidth=2)
ax1.plot(edge_only, label="Edge-only", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2)
ax2.plot(edge_only, label="Edge-only", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2)
ax1.plot(cloud_only, label="Cloud-only", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
ax2.plot(cloud_only, label="Cloud-only", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)

fig.suptitle("Multi-User Diff Method")
ax1.set_ylim(17000, 19826.23939977911 * 1.2 + 1000)  # most of the data
ax2.set_ylim(780, 1600)  # outliers only

# 隐藏两个坐标轴系列之间的横线
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()

# 隐藏y轴刻度
ax1.tick_params(axis='x', length=0)
# ax2.xaxis.tick_bottom()

# 添加网格线
ax1.grid(ls='--', alpha=0.5, linewidth=1)
ax2.grid(ls='--', alpha=0.5, linewidth=1)

# 创建轴断刻度线，d用于调节其偏转角度
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)

ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# plt.legend(loc=1)

plt.xlabel('Episode')
plt.ylabel('Cost')
plt.legend()
# plt.savefig('1.broken_yaxis.png', dpi=600, bbox_inches='tight')
plt.show()
