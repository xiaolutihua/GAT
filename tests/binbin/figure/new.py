import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

# 创建用于绘制图片的30个处于0到0.2之间的随机数据点
pts = np.random.rand(30) * 0.2

# 创建第二组数据
pts2 = np.random.rand(30) * 0.2

# 在原始数据点中创建两个离群数据点
pts[[3, 14]] += 0.8

# 创建两个绘图坐标轴；调整两个轴之间的距离，即轴断点距离
# 设置两个坐标轴的高度比为 1:9
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 9]})
fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# 将用相同的绘图数据，在两个轴上绘制折线图
ax1.plot(pts)
ax2.plot(pts)

# 创建第二组数据的额外轴
ax2_right = ax2.twinx()
ax2_right.plot(pts2, 'r-')  # 使用红色线条绘制第二组数据

# 调整两个y轴的显示范围
ax1.set_ylim(.78, 1.)  # outliers only
ax2.set_ylim(0, .22)   # most of the data for the left y-axis
ax2_right.set_ylim(0, .22)  # same range for the right y-axis as the left y-axis

# 隐藏两个坐标轴系列之间的横线
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top

# 隐藏y轴刻度
ax1.tick_params(axis='x', which='both', length=0)
ax2.xaxis.tick_bottom()

# 添加网格线
ax1.grid(ls='--', alpha=0.5, linewidth=1)
ax2.grid(ls='--', alpha=0.5, linewidth=1)

# 创建轴断刻度线，d用于调节其偏转角度
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# 添加整个图形的标题
fig.suptitle('Broken Y Axis with Dual Y Axes Example', fontsize=16)

plt.savefig('broken_yaxis_dual.png', dpi=600, bbox_inches='tight')
plt.show()
