import os
import json
import numpy as np

folder_path = f'{os.path.dirname(__file__)}/../../results/ppo/cycle/without_bc'

file_list = os.listdir(folder_path)
count = 0
# 遍历文件列表
for file_name in file_list:
    if file_name.endswith('.json'):  # 确保文件是JSON文件
        count += 1
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            loaded_data = json.load(file)
            costs = loaded_data["cost"]
            overtime = loaded_data["overtime"]
            # 刪除超时的
            for time in overtime:
                if time:
                    pass
