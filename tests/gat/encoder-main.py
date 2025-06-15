import time
from typing import List
from gat.graph_attention import StructureLoss, Coder, Encoder, Decoder
import torch
from binbin.gym_env import TaskpoolState
from utils import pickle_load, sparse_matrix_2_adj
import os
from tqdm import *

path = f"{os.path.dirname(__file__)}/../../graph_datas"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称

n_epoch = 100

n_node_feature = 5
n_task_feature_dim = 6

if __name__ == '__main__':

    graphs: List["TaskpoolState"] = []
    count = 0
    for file in files:  # 遍历文件夹
        if count >= 500:
            break
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            file_path = f"{path}/{file}"
            graph: "TaskpoolState" = pickle_load(file_path)
            graphs.append(graph)
        count += 1
    print("------------------数据加载完成----------------")
    print(f"共有{len(graphs)}")
    # 定义encoder, decoder, 损失函数, 优化器

    encoder = Encoder(n_node_feature, 10, n_task_feature_dim, 0.2, 0.2, 5, 3)
    decoder = Decoder(n_task_feature_dim, 10, n_node_feature, 0.2, 0.2, 5, 3)

    feature_loss = torch.nn.MSELoss()
    structure_loss = StructureLoss()

    optimal1 = torch.optim.Adam(encoder.parameters(), lr=0.001)
    optimal2 = torch.optim.Adam(decoder.parameters(), lr=0.001)

    count = 0
    # train
    for _ in range(n_epoch):
        f_loss_sum = 0
        s_loss_sum = 0
        for graph in graphs:
            x = torch.tensor(graph.tasks, dtype=torch.float32)
            # 将 spares_matrix 转化为 adjust_matrix
            adj = sparse_matrix_2_adj(graph.adj)

            x_hat = encoder(x, adj.T)
            x_dot = decoder(x_hat, adj.T)

            # structure_loss
            s_loss = structure_loss(x_hat, adj.T)
            # print("s_loss", s_loss)

            # feature_loss
            f_loss = feature_loss(x, x_dot)
            # print("f_loss", f_loss)

            s_loss_sum += s_loss.item()
            f_loss_sum += f_loss.item()

            optimal1.zero_grad()
            optimal2.zero_grad()

            s_loss.backward(retain_graph=True)
            f_loss.backward()

            optimal1.step()
            optimal2.step()
            # for name, parms in encoder.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
        print("f_loss_mean", f_loss_sum / len(graphs))  # 特征损失, 输入input 与 decoder的输出前后的差异
        print("s_loss_mean", s_loss_sum / len(graphs))  # 结构损失, 邻居节点结构相差不大
        print("=======================================")

        # save model parameter
        save_path = f"{os.path.dirname(__file__)}/../../graph_model" \
                    f"/encoder-checkpoint-{n_task_feature_dim}-{count}.model"
        print("saving model -> {}".format(save_path))
        # torch.save(encoder.state_dict(), save_path) # 保存参数
        torch.save(encoder, save_path)
        count += 1

    print("the {} epoch have finished!".format(n_epoch))
