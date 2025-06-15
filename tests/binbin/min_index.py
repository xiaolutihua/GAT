a = [11, 11, 2, 3, 4, 2]
b = [0.2, 0.3, 0.3, 0.2, 0.1, 0.1]

min_a_index = a.index(min(a))  # 获取在列表a中最小值的索引
min_b_index = [i for i, val in enumerate(b) if val == min(b)]  # 获取在列表b中最小值的索引

result = min(set(min_b_index) & set([min_a_index]))  # 取交集找到符合条件的索引

print(result)
