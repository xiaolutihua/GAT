import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

cycle_reduce_rate = [0.0815562477215017, 0.11252807674145587, 0.13587501985687667, 0.18456118922466913,
                     0.23037327813372185, 0.25759044327635783, 0.2946361386561471, 0.30627661264415035,
                     0.32522558912936006, 0.3922273021729137, 0.43095514144503616, 0.4753771399833909,
                     0.49053723678222655, 0.6094900111589197, 0.7320414707232795, 0.7981743966704699,
                     0.8643073226176603, 0.920992687715252]
y_without_gat = [0.48632961305965416, 0.48743812573075085, 0.5538187982466863, 0.48383045419522586, 0.4512339930870229,
                 0.47786063623247665, 0.4103209888613081, 0.3935990486105344, 0.3835816817130941, 0.40453821218612074,
                 0.2918574021113004, 0.2396218694178845, 0.29527947144842076, 0.217093340314575, 0.16653926496061477,
                 0.12392019858648709, 0.07252981622435296, 0.03833293551630512]

y_normal = [0.6835859166082517, 0.6770868340690744, 0.6345139652672859, 0.600948408067812, 0.5646388098960748,
            0.5194551745928161, 0.5042157657764837, 0.4945049162398786, 0.5114829162791623, 0.4698082366240754,
            0.4145384302232643, 0.33282922268996934, 0.4383364847705693, 0.3290967355831331, 0.2280155210150697,
            0.1710784048773262, 0.14342636748150822, 0.013428803721964746]


f = interp1d(cycle_reduce_rate, y_without_gat, kind='cubic')
smoothed_x = np.linspace(min(cycle_reduce_rate), max(y_without_gat), 100)
smoothed_y_without_gat = f(smoothed_x)

f = interp1d(cycle_reduce_rate, y_normal, kind='cubic')
smoothed_x = np.linspace(min(cycle_reduce_rate), max(y_normal), 100)
smoothed_y_normal = f(smoothed_x)

# 创建数据
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 6]
y2 = [1, 2, 4, 5, 7]
y3 = [3, 1, 6, 4, 8]
y4 = [2, 5, 3, 6, 9]

# 创建第一个子图
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
# plt.plot(cycle_reduce_rate, y_without_gat, label='without_gat', color='blue')
# plt.plot(cycle_reduce_rate, y_normal, label='normal', color='red')
#
# plt.xlabel('cycle reduce rate')
# plt.ylabel('optimal rate')
# plt.legend()

# 创建第二个子图
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(smoothed_x, smoothed_y_without_gat, label='disable_encoder', color='blue', linestyle="-.")
plt.plot(smoothed_x, smoothed_y_normal, label='enable_encoder', color='red', linestyle="-")
plt.xlabel('Cycle Reduce Rate')
plt.ylabel('Optimal Rate')
plt.title("Ablation Experiments For Encoder")
plt.legend()

# 调整子图布局
# plt.tight_layout()

# 显示图形
plt.show()
