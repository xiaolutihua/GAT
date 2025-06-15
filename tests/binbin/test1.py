# import numpy as np
#
# test = [i for i in range(10)]
# for _ in range(3):
#     flag = True
#     if flag:
#         flag = False
#         test = np.array(test)
#         print("flag", test)
#     print(type(test))

# import json
# import matplotlib.pyplot as plt
#
# with open(
#     "D:\\BaiduSyncdisk\\DAG-MEC\\experiment\\results\\ppo\\algo\\learn_rate\\reward\\C_1_5_2550_0.15_284fc396-ecd7-11ee-9277-e86f38617572.json",
#     'r') as file:
#     loaded_data = json.load(file)
#     costs = loaded_data["cost"]
#     plt.plot(range(len(costs)), costs)
#     plt.show()


import matplotlib.pyplot as plt

min_cost = [
    [712.6232231951969, 0.98, 1e-05],
    [720.700398316504, 0.9, 1e-05],
    [706.0933939888421, 0.84, 1e-05],
    [716.5835134092251, 0.94, 1e-05],
    [750.271876502661, 0.92, 1e-05],
    [822.205556144221, 0.96, 1e-05],
    [685.745834241388, 0.88, 1e-05],
    [800.6922337400165, 0.84, 1e-05],
    [702.828666125219, 0.8200000000000001, 1e-05],
    [755.6701059766482, 1.0, 1e-05],
    [688.264375212686, 0.88, 1e-05],
    [566.9682164048962, 0.8200000000000001, 1e-05],
    [713.8712700387088, 0.94, 1e-05],
    [689.8062763131222, 0.96, 1e-05],
    [783.5771972412804, 1.0, 1e-05],
    [716.0539262803176, 0.92, 1e-05],
    [798.5576678350003, 0.86, 1e-05],
    [645.7770878795536, 0.86, 1e-05],
    [850.7963800404406, 0.98, 1e-05],
    [642.6119817928876, 0.9, 1e-05],
]
costs = []
gammas = []
for experiment in min_cost:
    if experiment[1] not in gammas:
        gammas.append(experiment[1])
        costs.append(experiment[0])
    else:
        index = gammas.index(experiment[1])
        costs[index] = min(costs[index], experiment[0])

print(costs)
print(gammas)

sorted_data = sorted(zip(gammas, costs))
sorted_gammas, sorted_costs = zip(*sorted_data)

plt.plot(sorted_gammas, sorted_costs)
plt.show()
