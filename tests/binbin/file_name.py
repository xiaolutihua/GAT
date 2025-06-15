import re

# 文件名
file_name = 'device3_5_2650_0.6_8e7cd95d-df76-11ee-94ed-581cf8a4c71f.json'

# 使用正则表达式提取数字
match = re.search(r'device(\d+)_', file_name)

if match:
    device_number = int(match.group(1))
    print(device_number)
else:
    print("No device number found in the file name.")
