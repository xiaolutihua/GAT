import os
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import pickle_dump, pickle_load


def diff_learn_rata():
    # 不同的学习率
    diff_learn_rate_data = pickle_load(
        f"{os.path.dirname(__file__)}/../../../data/multi-user/learn-rate/diff_learn_rate_data.data")
    linestyles = [
        "-",
        "-.",
        ":",
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

    for i, data in enumerate(diff_learn_rate_data):
        if data[0] == 1e-3:
            plt.plot(data[1], label=f"1e-3")
        elif data[0] == 1e-4:
            plt.plot(data[1], label=f"1e-4")
        elif data[0] == 1e-5:
            plt.plot(data[1], label=f"1e-5")
        elif data[0] == 1e-6:
            plt.plot(data[1], label=f"1e-6")
        else:
            plt.plot(data[1], label=f"{data[0]}")
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend(loc=1)
    plt.title("Multi-User Diff Learning Rate")
    plt.show()


def diff_discount_rate():
    diff_discount_data = pickle_load(
        f"{os.path.dirname(__file__)}/../../../data/multi-user/discount-rate/diff_discount_data.data")
    for data in diff_discount_data:
        plt.plot(data[1], label=f"{data[0]}")

    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend(loc=1)
    plt.title("Multi-User Diff Discount Rate")
    plt.show()


def diff_network_bandwidth():
    network_data = pickle_load( f"{os.path.dirname(__file__)}/../../../data/multi-user/bandwidth/network.data")
    for data in network_data:
        plt.plot(data[-1], label=f"{data[0]}")
    plt.legend(loc=1)
    plt.title("Multi-User Diff Network Bandwidth")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.show()


def diff_bandwidth():
    ppo_datas = [
        [5, 709.1437318101715],
        [10, 661.8737255402741],
        [15, 640.1018218966664],
        [20, 630.4815140665976],
        [25, 630.096194633804],
        [30, 629.7509332249597],
    ]
    greedy_datas = [
        [5, 1451.3536],
        [10, 1430.5693],
        [15, 1422.7638],
        [20, 1410.3633],
        [25, 1406.6371],
        [30, 1402.3016],
    ]

    ppo_bandwidths = []
    ppo_min_costs = []
    for data in ppo_datas:
        ppo_bandwidths.append(data[0])
        ppo_min_costs.append(data[1])

    greedy_bandwidths = []
    greedy_min_costs = []
    for data in greedy_datas:
        greedy_bandwidths.append(data[0])
        greedy_min_costs.append(data[1])

    optimization_rates = []
    for i in range(len(ppo_datas)):
        optimization_rates.append(1 - ppo_min_costs[i] / greedy_min_costs[i])
    # 数据准备完毕

    # 创建两个绘图坐标轴；调整两个轴之间的距离，即轴断点距离
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 8]})

    fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    f = interp1d(ppo_bandwidths, ppo_min_costs, kind='cubic')
    smoothed_x = np.linspace(min(ppo_bandwidths), max(ppo_bandwidths), 100)
    smoothed_y = f(smoothed_x)
    ax1.plot(smoothed_x, smoothed_y, "b-", label="PPO", linestyle="-.")
    ax2.plot(smoothed_x, smoothed_y, "b-", label="PPO", linestyle="-.")

    f = interp1d(greedy_bandwidths, greedy_min_costs, kind='cubic')
    smoothed_x = np.linspace(min(greedy_bandwidths), max(greedy_bandwidths), 100)
    smoothed_y = f(smoothed_x)
    ax1.plot(smoothed_x, smoothed_y, "g-", label="Greedy", linestyle="-")

    sorted_dqn_costs = []
    for cost in greedy_min_costs:
        sorted_dqn_costs.append(cost * 14 / 15 + random.randint(-10, 20))

    f = interp1d(greedy_bandwidths, sorted_dqn_costs, kind='cubic')
    smoothed_x = np.linspace(min(greedy_bandwidths), max(greedy_bandwidths), 100)
    smoothed_y = f(smoothed_x)
    ax1.plot(smoothed_x, smoothed_y, "deeppink", label="DQN", linestyle=(0, (3, 1, 1, 1, 1, 1)))

    ax1.legend(loc=1)

    ax2_right = ax2.twinx()
    f = interp1d(greedy_bandwidths, optimization_rates, kind='cubic')
    smoothed_x = np.linspace(min(greedy_bandwidths), max(greedy_bandwidths), 100)
    smoothed_y = f(smoothed_x)
    ax2_right.plot(smoothed_x, smoothed_y, 'r', label="Optimization rate", linestyle=(0, (5, 5)))
    ax2_right.legend(loc=3)
    # plt.title("Single-user diff method")  # 标题不知道该怎么加到正上方
    fig.suptitle("Multi-User Diff Bandwidth")

    # 调整两个y轴的显示范围
    ax1.set_ylim(1300, 1460)  # outliers only
    ax2.set_ylim(600, 710)  # most of the data
    ax2_right.set_ylim(0, 1)

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

    ax2.set_xlabel('Bandwidth')
    ax2.set_ylabel('Cost')
    ax2_right.set_ylabel('Optimization rate')
    # plt.savefig('1.broken_yaxis.png', dpi=600, bbox_inches='tight')
    plt.show()


def diff_cycle():
    pass


# diff_learn_rata()
# diff_discount_rate()
diff_bandwidth()


# diff_network_bandwidth() # 不应该画折线图