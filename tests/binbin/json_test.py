import json

# # 要保存的字典
data = {'name': 'Alice', 'age': 25, 'city': 'New York'}
#
# # 将字典写入到文件
with open('data.json', 'w') as file:
    json.dump(data, file)


# 从文件中读取字典
with open('data.json', 'r') as file:
    loaded_data = json.load(file)

print(loaded_data)
print(loaded_data["name"])
