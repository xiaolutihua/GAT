import torch
from tqdm import tqdm

from graph import Graph
from graph_attention import StructureLoss, Coder
from utils import get_gv_root, get_all_gv_files_in_data_dir

n_epoch = 100

n_node_feature = 4

if __name__ == '__main__':

    graphs = []
    gv_files = get_all_gv_files_in_data_dir(get_gv_root())
    print("Loading gv files...")
    for i in tqdm(range(len(gv_files))):
        gv_file = gv_files[i]
        graph = Graph.from_gv_file(gv_file)
        graphs.append(graph)

    # 定义encoder, decoder, 损失函数, 优化器

    encoder = Coder(n_node_feature, 10, n_node_feature, 0.2, 0.2, 5, 3)
    decoder = Coder(n_node_feature, 10, n_node_feature, 0.2, 0.2, 5, 3)

    feature_loss = torch.nn.MSELoss()
    structure_loss = StructureLoss()

    optimal1 = torch.optim.Adam(encoder.parameters(), lr=0.001)
    optimal2 = torch.optim.Adam(decoder.parameters(), lr=0.001)

    # train
    print('Training...')
    for _ in (pbar := tqdm(range(n_epoch))):
        f_loss_sum = 0
        s_loss_sum = 0
        for graph in graphs:
            x, adj = graph.to_tensor()

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
        pbar.set_description(f"f_loss_mean={f_loss_sum / len(graphs)} s_loss_mean={s_loss_sum / len(graphs)}")

    print("The {} epoch have finished!".format(n_epoch))

    # save model parameter
    print("Saving model...")
    torch.save(encoder.state_dict(), "encoder.pt")
    torch.save(decoder.state_dict(), "decoder.pt")
