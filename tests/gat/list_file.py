import os

path = "C:\\Users\\hyb\\Desktop\\DAG-MEC\\test\\simulator\\graph_datas"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
s = []
for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        print(file)
print(s)  # 打印结果
