import os
from utils import pickle_load
from binbin.gym_env import TaskpoolState

print("准备导出graph图")

graph_data_path = os.path.dirname(__file__)

print(graph_data_path)
# 加载数据
graph_data: "TaskpoolState" = pickle_load(
    "C:\\Users\\hyb\\Desktop\\DAG-MEC\\test\\simulator\\graph_datas\\9a01c1b5-c42b-11ee-a99c-e86f38617572.graph")

print(graph_data.adj)
print(graph_data.tasks)

# 训练encoder



