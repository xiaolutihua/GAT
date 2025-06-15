import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# 数据
average_cycle_reduce_rates = [0.11252807674145587, 0.0815562477215017, 0.30627661264415035, 0.25759044327635783, 0.18456118922466913, 0.2946361386561471, 0.13587501985687667, 0.23037327813372185, 0.3922273021729137, 0.32522558912936006, 0.43095514144503616]
optimal_rate = [0.32291316593092556, 0.31641408339174826, 0.5054950837601214, 0.4805448254071839, 0.399051591932188, 0.49578423422351625, 0.3654860347327142, 0.4353611901039252, 0.5301917633759246, 0.48851708372083774, 0.6229335326864464]

# 对average_cycle_reduce_rates进行排序，并根据排序结果重新排列optimal_rate
sorted_data = sorted(zip(average_cycle_reduce_rates, optimal_rate))
sorted_average_cycle_reduce_rates, sorted_optimal_rate = zip(*sorted_data)

# 创建插值函数
f = interp1d(sorted_average_cycle_reduce_rates, sorted_optimal_rate, kind='cubic')

# 生成平滑处理后的数据点
smoothed_x = np.linspace(min(sorted_average_cycle_reduce_rates), max(sorted_average_cycle_reduce_rates), 500)
smoothed_y = f(smoothed_x)

# 创建图形
plt.plot(smoothed_x, smoothed_y, marker='o', linestyle='-', linewidth=0.5)  # 调整线条粗细为1.5

# 添加标题和标签
plt.title('Optimal Rate vs Average Cycle Reduce Rates (Smoothed)')
plt.xlabel('Average Cycle Reduce Rates')
plt.ylabel('Optimal Rate')

# 显示图形
plt.show()
