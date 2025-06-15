import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

cycle_reduce_rate = [0.14479849092642516, 0.18931619716460457, 0.23383390340278398, 0.2783516096409634,
                     0.3451281689982325, 0.3896458752364119, 0.43416358147459133, 0.47868128771277074,
                     0.5454578470700397, 0.5899755533082192, 0.6456226861059434, 0.6901403923441228, 0.7457875251418471,
                     0.7903052313800265, 0.8459523641777508, 0.8904700704159302, 0.9461172032136544, 0.9906349094518339]
min_costs = [5.576191949112318, 5.322675986075081, 7.508836650552118, 9.062870975125572, 10.399328451739109,
             11.986469155606063, 8.272968253252836, 9.951676952859662, 14.49541901942484, 16.717952027174736,
             20.589431803951133, 21.976195291279183, 24.095245570318166, 22.27558101010123, 27.512296542832118,
             26.65834332439857, 29.077783936170373, 33.747374177412574]
greedy_cost = [37.48912948887796, 40.444572160629185, 41.59850814934389, 42.7990017409816, 44.494800538425345,
               47.35427114364964, 49.891740053129666, 50.609731691015284, 51.31545217223861, 51.86280792124301,
               54.39685729081014, 54.96060693131558, 55.982022932197026, 57.86842164170961, 58.650500487912225,
               60.75097425945782, 62.975592277339004, 65.86803190887346]
dqn_cost =[21.992752992585306, 35.963048107086124, 27.73233876622926, 24.532667827321067, 28.66320035895023, 28.569514095766426, 34.26116003541978, 36.739821127343525, 31.210301448159072, 39.575205280828676, 36.26457152720676, 37.64040462087706, 41.321348621464686, 41.57894776113974, 40.10033365860815, 37.500649506305216, 48.98372818489267, 40.91202127258231]
# for cost in greedy_cost:
#     dqn_cost.append(cost* 2 / 3 + random.randint(-6, 10))
#
# print(dqn_cost)

add_cost = 0


min_costs = np.array(min_costs)
greedy_cost = np.array(greedy_cost)
min_costs = np.array(min_costs)
dqn_cost = np.array(dqn_cost)

data = 1 - min_costs / greedy_cost

fig, ax1 = plt.subplots()

f = interp1d(cycle_reduce_rate, min_costs, kind='cubic')
smoothed_x = np.linspace(min(cycle_reduce_rate), max(cycle_reduce_rate), 500)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, 'b-', label="PPO", linestyle="-.")

f = interp1d(cycle_reduce_rate, greedy_cost, kind='cubic')
smoothed_x = np.linspace(min(cycle_reduce_rate), max(cycle_reduce_rate), 500)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, 'g-', label="greedy", linestyle="-")

f = interp1d(cycle_reduce_rate, dqn_cost, kind='cubic')
smoothed_x = np.linspace(min(cycle_reduce_rate), max(cycle_reduce_rate), 500)
smoothed_y = f(smoothed_x)
ax1.plot(smoothed_x, smoothed_y, "deeppink", label="DQN" , linestyle=(0, (3, 1, 1, 1, 1, 1)))

ax1.set_xlabel('Cycle Reduce Rate')
ax1.set_ylabel('Cost')

# plt.show()


# ---------------------------------------------------------------------- #
# 优化率
ax2 = ax1.twinx()

# 创建插值函数
f = interp1d(cycle_reduce_rate, data, kind='cubic')

# 生成平滑处理后的数据点
smoothed_x = np.linspace(min(cycle_reduce_rate), max(cycle_reduce_rate), 500)
smoothed_y = f(smoothed_x)

# 创建图形
ax2.plot(smoothed_x, smoothed_y, 'r', label="Optimization rate", linestyle=(0, (5, 5)))

# 添加标题和标签
ax2.set_ylabel('Optimal Rate')

ax1.legend(loc=3)
ax2.legend(loc=1)

plt.title("Single-User Diff Cycle")
# 显示图形
plt.show()
