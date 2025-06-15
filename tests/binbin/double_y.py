import matplotlib.pyplot as plt

# 创建数据
x = range(10)
y1 = [i**2 for i in x]  # y1是x的平方
y2 = [i**0.5 for i in x]  # y2是x的平方根

# 创建一个新的figure和axis对象
fig, ax1 = plt.subplots()

# 绘制第一组数据，使用ax1的y轴（默认左边）
ax1.plot(x, y1, 'g-', label='y1 = x^2')
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')

# 创建第二个y轴，共享相同的x轴
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-', label='y2 = sqrt(x)')
ax2.set_ylabel('Y2 data', color='b')

# 可选：设置legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图表
plt.show()
