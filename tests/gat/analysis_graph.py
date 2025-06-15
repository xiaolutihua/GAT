import os
from typing import List
import matplotlib as plt
from utils import pickle_load
from binbin.gym_env import TaskpoolState

path = "C:\\Users\\hyb\\Desktop\\DAG-MEC\\test\\simulator\\graph_datas"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称

if __name__ == '__main__':
    compute_sizes = []
    ram_sizes = []
    translate_sizes = []
    result_sizes = []
    tolerate_times = []
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            file_path = f"{path}/{file}"
            graph: "TaskpoolState" = pickle_load(file_path)
            for task in graph.tasks:
                compute_sizes.append(task[0])
                translate_sizes.append(task[1])
                ram_sizes.append(task[2])
                result_sizes.append(task[3])
                tolerate_times.append(task[4])
    print("------------------数据加载完成----------------")

    print("compute_size: ", max(compute_sizes), min(compute_sizes))
    print("ram_sizes: ", max(ram_sizes), min(ram_sizes))
    print("translate_sizes: ", max(translate_sizes), min(translate_sizes))
    print("result_sizes: ", max(result_sizes), min(result_sizes))
    print("tolerate_times: ", max(tolerate_times), min(tolerate_times))

    print(
        1241464162071,
        49860300,
        1028010,
        91136,
        4100
    )

    print(
        1.3e12,
        5e7,
        1.1e6,
        1e5,
        5e3
    )

