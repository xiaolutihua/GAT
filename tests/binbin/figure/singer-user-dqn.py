import os
import random

import numpy as np
import matplotlib.pyplot as plt
from utils import pickle_load, pickle_dump


def cast_to_cost(args):
    overtimes = []
    for i, reward in enumerate(args):
        if reward <= -10000:
            args[i] += 10000
            overtimes.append(True)
        else:
            overtimes.append(False)
        args[i] = abs(args[i])
    return args, overtimes


def diff_method():
    dqn_rewards = pickle_load(f"{os.path.dirname(__file__)}/../../../data/single-user/method/dqn-reward.data")
    ppo_rewards = pickle_load(f"{os.path.dirname(__file__)}/../../../data/single-user/method/ppo-reward.data")

    print(ppo_rewards)
    dqn_rewards, dqn_overtimes = cast_to_cost(dqn_rewards)
    ppo_rewards, ppo_overtimes = cast_to_cost(ppo_rewards)

    print(len(dqn_rewards))
    print(len(ppo_rewards))

    greedy_rewards = [124.91231696280651 for i in range(400)]
    edge_only_rewards = [501.91231696280651 for i in range(400)]
    cloud_only_rewards = [602.4 for i in range(400)]

    # 创建两个绘图坐标轴；调整两个轴之间的距离，即轴断点距离
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 8]})

    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    ax1.plot(ppo_rewards, label="PPO")
    ax2.plot(ppo_rewards, label="PPO")
    ax1.plot(dqn_rewards, label="DQN")
    ax2.plot(dqn_rewards, label="DQN")
    ax1.plot(greedy_rewards, label="Greedy")
    ax2.plot(greedy_rewards, label="Greedy")
    # https://zhuanlan.zhihu.com/p/157350270 ,j
    ax1.plot(edge_only_rewards, label="Edge-Only")
    ax2.plot(edge_only_rewards, label="Edge-Only")
    # ax2.plot(cloud_only_rewards, label="Cloud-Only")
    # ax1.plot(cloud_only_rewards, label="Cloud-Only")
    ax1.legend(loc=1)
    # plt.title("Single-user diff method")  # 标题不知道该怎么加到正上方
    fig.suptitle("Single-User Diff Method")
    # 调整两个y轴的显示范围
    ax1.set_ylim(460, 560)  # outliers only
    ax2.set_ylim(0, 150)  # most of the data

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
    # plt.savefig('1.broken_yaxis.png', dpi=600, bbox_inches='tight')
    plt.show()


def diff_discount_rate():
    img_data = pickle_load(f"{os.path.dirname(__file__)}/../../../data/single-user/discount-rate/img-data.data")
    for data in img_data:
        plt.plot(data[1], label=f"{data[0]}")

    plt.title("Single-User Diff Discount Rate")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()


def diff_bandwidth():
    pass


def diff_learn_rate():
    tmp_data = pickle_load(f"{os.path.dirname(__file__)}/../../../data/single-user/learn-rate/tmp-data.data")

    for data in tmp_data:
        plt.plot(data[-2], label=f"{data[2]}")
        # if data[2] == 1e-5:
        #     print(data[-2])

    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.title("Single-User Diff Learning Rate")
    plt.legend()
    plt.show()


diff_method()
diff_discount_rate()
diff_learn_rate()
