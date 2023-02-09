import networkx as nx
from torch_scatter import scatter
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix, add_remaining_self_loops, k_hop_subgraph, degree, to_networkx
from scipy.sparse import csr_matrix
import random
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length
import os


# def propagate(x, edge_index, edge_weight=None):
#     """ feature propagation procedure: sparsematrix
#     """
#     edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

#     # calculate the degree normalize term
#     row, col = edge_index
#     deg = degree(col, x.size(0), dtype=x.dtype)
#     deg_inv_sqrt = deg.pow(-0.5)
#     # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
#     if(edge_weight == None):
#         edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#     # normalize the features on the starting point of the edge
#     out = edge_weight.view(-1, 1) * x[row]

#     return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate(x, edge_index, edge_weight):
    """ feature propagation procedure: sparsematrix
    """
    row, col = edge_index

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, col, dim=0, dim_size=x.size(0), reduce='add')


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def index_to_mask(train_index, val_index, test_index, size):
    train_mask = torch.zeros(size, dtype=torch.bool)
    val_mask = torch.zeros(size, dtype=torch.bool)
    test_mask = torch.zeros(size, dtype=torch.bool)

    train_mask[train_index] = 1
    val_mask[val_index] = 1
    test_mask[test_index] = 1

    return train_mask, val_mask, test_mask


def subgraph_extract(data, train_num, k_hop):
    train_node_idx = torch.tensor(np.random.choice(torch.where(
        data.train_mask == True)[0].numpy(), train_num, replace=False))

    subnodes, sub_edge_index, node_mapping, edge_mapping = k_hop_subgraph(
        train_node_idx, k_hop, data.edge_index, relabel_nodes=True)

    sub_x = data.x[subnodes]
    sub_train_mask, sub_val_mask, sub_test_mask = data.train_mask[
        subnodes], data.val_mask[subnodes], data.test_mask[subnodes]

    sub_y = data.y[subnodes]

    return sub_x, sub_y, sub_edge_index, sub_train_mask, sub_val_mask, sub_test_mask, subnodes


def shortest_path(data, subnodes):
    G = to_networkx(data)
    p_l = torch.zeros((len(subnodes), data.x.size(0)))

    for i in range(len(subnodes)):
        lengths = single_source_shortest_path_length(
            G, source=subnodes[i].item())
        for key in lengths:
            if(lengths[key] != 0):
                p_l[i, key] = 1 / lengths[key]
            else:
                p_l[i, key] = 1

    p_l = p_l.t()

    return p_l


def cal_jc(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0])
    adj_matrix[edge_index[0], edge_index[1]] = 1

    edge_weight = torch.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)

        neighbors_adj = adj_matrix[neighbors]

        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t())
        nei_nei_cup = neighbors_adj.sum(
            dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1)

        jacard_simi = (nei_nei_cap / (nei_nei_cup - nei_nei_cap)).mean(dim=1)

        edge_weight[i, neighbors] = jacard_simi

    edge_weight = edge_weight[edge_index[1], edge_index[0]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/cir_jc.pt')

    return edge_weight


def cal_sc(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0])
    adj_matrix[edge_index[0], edge_index[1]] = 1

    edge_weight = torch.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)

        neighbors_adj = adj_matrix[neighbors]

        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t())
        nei_nei_cup = neighbors_adj.sum(
            dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1)

        sc_simi = (nei_nei_cap / ((neighbors_adj.sum(dim=1) *
                                   neighbors_adj.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[i, neighbors] = sc_simi

    edge_weight = edge_weight[edge_index[1], edge_index[0]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/cir_sc.pt')

    return edge_weight


def cal_lhn(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0])
    adj_matrix[edge_index[0], edge_index[1]] = 1

    edge_weight = torch.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)

        neighbors_adj = adj_matrix[neighbors]

        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t())
        nei_nei_cup = neighbors_adj.sum(
            dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1)

        lhn_simi = (nei_nei_cap / ((neighbors_adj.sum(dim=1) *
                                    neighbors_adj.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[i, neighbors] = lhn_simi

    edge_weight = edge_weight[edge_index[1], edge_index[0]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/cir_lhn.pt')

    return edge_weight


def cal_co(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0])
    adj_matrix[edge_index[0], edge_index[1]] = 1

    edge_weight = torch.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)

        neighbors_adj = adj_matrix[neighbors]

        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t())
        nei_nei_cup = neighbors_adj.sum(
            dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1)

        co_simi = nei_nei_cap.mean(dim=1)

        edge_weight[i, neighbors] = co_simi

    edge_weight = edge_weight[edge_index[1], edge_index[0]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/cir_co.pt')

    return edge_weight
