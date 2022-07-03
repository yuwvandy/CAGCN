import numpy as np
import random
import torch
import scipy.sparse as sp
from torch import nn
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter_max, scatter_add
import seaborn as sns
import matplotlib.pyplot as plt
import os


def neg_sample_before_epoch(train_cf, clicked_set, args):
    neg_cf = np.random.randint(
        args.n_users, args.n_users + args.n_items, (train_cf.shape[0], args.K))

    for i in range(train_cf.shape[0]):
        # neg items will not include in user_clicked_set
        user_clicked_set = clicked_set[train_cf[i, 0]]

        for j in range(args.K):
            while(neg_cf[i, j] in user_clicked_set):
                neg_cf[i, j] = np.random.randint(
                    args.n_users, args.n_users + args.n_items)

    return neg_cf


def batch_to_gpu(batch, device):
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def knn_adj(adj_sp_norm, args):
    adj_sp_norm = adj_sp_norm.to_dense()
    top_adj_sp_norm, _ = torch.topk(adj_sp_norm, args.knn)
    low, high = top_adj_sp_norm[:, -1], top_adj_sp_norm[:, 0]
    mask = ((adj_sp_norm >= low.unsqueeze(1)) *
            (adj_sp_norm <= high.unsqueeze(1))).float()
    adj_sp_norm = adj_sp_norm * mask
    adj_sp_norm = torch.triu(adj_sp_norm, diagonal=1)

    edge_index = adj_sp_norm.nonzero()
    adj_sp_norm = torch.sparse.FloatTensor(
        edge_index.t(), adj_sp_norm[edge_index[:, 0], edge_index[:, 1]], (args.n_users + args.n_items, args.n_users + args.n_items))

    return adj_sp_norm


def ratio(train_cf, n_users):
    user_link_num = torch.tensor(
        [(train_cf[:, 0] == i).sum() for i in range(n_users)])

    link_ratio = n_users / user_link_num
    link_ratio = link_ratio / link_ratio.sum()

    return link_ratio[train_cf[:, 0]]


def cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    pos_scores = torch.sum(
        torch.mul(user_embs, pos_item_embs), axis=1)

    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(
        dim=1), neg_item_embs), axis=-1)

    bpr_loss = torch.mean(
        torch.log(1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1)))

    # bpr_loss = (link_ratios*torch.log(1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1))).mean()

    return bpr_loss


def cal_l2_loss(user_embs, pos_item_embs, neg_item_embs, batch_size):
    return 0.5 * (user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / batch_size


def softmax(src, index, num_nodes):
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()

    # out = src #method 2

    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


def co_ratio_deg_user_jacard(adj_sp_norm, edge_index, degree, args):
    user_item_graph = adj_sp_norm.to_dense(
    )[:args.n_users, args.n_users:].cpu()
    user_item_graph[user_item_graph > 0] = 1

    item_user_graph = user_item_graph.t()

    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = user_item_graph[:, i].nonzero().squeeze(-1)

        items = user_item_graph[users]
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        jacard_simi = (user_user_cap / (user_user_cup -
                                        user_user_cap)).mean(dim=1)

        edge_weight[users, i + args.n_users] = jacard_simi

    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        jacard_simi = (item_item_cap / (item_item_cup -
                                        item_item_cap)).mean(dim=1)

        edge_weight[items + args.n_users, i] = jacard_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_jc.pt')

    return edge_weight


def co_ratio_deg_user_jacard_sp(adj_sp_norm, edge_index, degree, args):
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = edge_index[0, edge_index[1] == i + args.n_users]
        items = torch.zeros([users.shape[0], args.n_items])

        for j, user in enumerate(users):
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1

        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        jacard_simi = (user_user_cap / (user_user_cup -
                                        user_user_cap)).mean(dim=1)

        edge_weight[users, i + args.n_users] = jacard_simi

    for i in range(args.n_users):
        items = edge_index[1, edge_index[0] == i]
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(
                args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        jacard_simi = (item_item_cap / (item_item_cup -
                                        item_item_cap)).mean(dim=1)

        edge_weight[items, i] = jacard_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_jc.pt')

    return edge_weight


def co_ratio_deg_user_common(adj_sp_norm, edge_index, degree, args):
    user_item_graph = adj_sp_norm.to_dense(
    )[:args.n_users, args.n_users:].cpu()
    user_item_graph[user_item_graph > 0] = 1

    item_user_graph = user_item_graph.t()

    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = user_item_graph[:, i].nonzero().squeeze(-1)

        items = user_item_graph[users]
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        common_simi = user_user_cap.mean(dim=1)

        edge_weight[users, i + args.n_users] = common_simi

    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        common_simi = item_item_cap.mean(dim=1)

        edge_weight[items + args.n_users, i] = common_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_co.pt')

    return edge_weight


def co_ratio_deg_user_common_sp(adj_sp_norm, edge_index, degree, args):
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = edge_index[0, edge_index[1] == i + args.n_users]
        items = torch.zeros([users.shape[0], args.n_items])

        for j, user in enumerate(users):
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1

        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        common_simi = user_user_cap.mean(dim=1)

        edge_weight[users, i + args.n_users] = common_simi

    for i in range(args.n_users):
        items = edge_index[1, edge_index[0] == i]
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(
                args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        common_simi = item_item_cap.mean(dim=1)

        edge_weight[items, i] = common_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_co.pt')

    return edge_weight


def co_ratio_deg_user_sc(adj_sp_norm, edge_index, degree, args):
    user_item_graph = adj_sp_norm.to_dense(
    )[:args.n_users, args.n_users:].cpu()
    user_item_graph[user_item_graph > 0] = 1

    item_user_graph = user_item_graph.t()

    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = user_item_graph[:, i].nonzero().squeeze(-1)

        items = user_item_graph[users]
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        sc_simi = (user_user_cap / ((items.sum(dim=1) *
                                     items.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[users, i + args.n_users] = sc_simi

    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        sc_simi = (item_item_cap / ((users.sum(dim=1) *
                                     users.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[items + args.n_users, i] = sc_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_sc.pt')

    return edge_weight


def co_ratio_deg_user_sc_sp(adj_sp_norm, edge_index, degree, args):
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = edge_index[0, edge_index[1] == i + args.n_users]
        items = torch.zeros([users.shape[0], args.n_items])

        for j, user in enumerate(users):
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1

        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        sc_simi = (user_user_cap / ((items.sum(dim=1) *
                                     items.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[users, i + args.n_users] = sc_simi

    for i in range(args.n_users):
        items = edge_index[1, edge_index[0] == i]
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(
                args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        sc_simi = (item_item_cap / ((users.sum(dim=1) *
                                     users.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[items, i] = sc_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_sc.pt')

    return edge_weight


def co_ratio_deg_user_lhn(adj_sp_norm, edge_index, degree, args):
    user_item_graph = adj_sp_norm.to_dense(
    )[:args.n_users, args.n_users:].cpu()
    user_item_graph[user_item_graph > 0] = 1

    item_user_graph = user_item_graph.t()

    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = user_item_graph[:, i].nonzero().squeeze(-1)

        items = user_item_graph[users]
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        lhn_simi = (user_user_cap / ((items.sum(dim=1) *
                                      items.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[users, i + args.n_users] = lhn_simi

    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        lhn_simi = (item_item_cap / ((users.sum(dim=1) *
                                      users.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[items + args.n_users, i] = lhn_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_lhn.pt')

    return edge_weight


def co_ratio_deg_user_lhn_sp(adj_sp_norm, edge_index, degree, args):
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = edge_index[0, edge_index[1] == i + args.n_users]
        items = torch.zeros([users.shape[0], args.n_items])

        for j, user in enumerate(users):
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1

        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        lhn_simi = (user_user_cap / ((items.sum(dim=1) *
                                      items.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[users, i + args.n_users] = lhn_simi

    for i in range(args.n_users):
        items = edge_index[1, edge_index[0] == i]
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(
                args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        lhn_simi = (item_item_cap / ((users.sum(dim=1) *
                                      users.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[items, i] = lhn_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_lhn.pt')

    return edge_weight


def visual_edge_weight(edge_weight, edge_index, degree, dataname, type):
    deg_1 = degree[edge_index[0, :]]
    deg_2 = degree[edge_index[1, :]]

    plt.figure()
    plt.scatter(deg_1.tolist(), edge_weight.tolist(), s=1, c='red', label=type)
    plt.scatter(deg_1.tolist(), 1 / deg_1**0.5 * 1 /
                deg_2**0.5, s=1, c='green', label='GCN', alpha=0.01)

    plt.xlabel('Degree of the head node', fontsize=15, fontweight='bold')
    plt.ylabel('Edge weight', fontsize=15, fontweight='bold')
    plt.legend()
    plt.savefig(os.getcwd() + '/result/' + dataname +
                '/edge_weight' + type + '_head' + '.pdf', dpi=50, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.scatter(deg_2.tolist(), edge_weight.tolist(), s=1, c='red', label=type)
    plt.scatter(deg_2.tolist(), 1 / deg_1**0.5 * 1 /
                deg_2**0.5, s=1, c='green', label='GCN', alpha=0.01)

    plt.xlabel('Degree of the tail node', fontsize=15, fontweight='bold')
    plt.ylabel('Edge weight', fontsize=15, fontweight='bold')
    plt.legend()
    plt.savefig(os.getcwd() + '/result/' + dataname +
                '/edge_weight' + type + '_tail' + '.pdf', dpi=50, bbox_inches='tight')
    plt.close()


def visual_node_rank(user_embs_aggr1, user_embs_aggr2, item_embs_aggr1, item_embs_aggr2, user_dict, topk, degree, n_users):
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    rating1 = torch.matmul(user_embs_aggr1, item_embs_aggr1.t())
    rating2 = torch.matmul(user_embs_aggr2, item_embs_aggr2.t())

    deg1, deg2 = [], []
    for i in range(n_users):
        color = []
        clicked_items = train_user_set[i] - n_users
        test_groundTruth_items = test_user_set[i] - n_users

        rating1[i, clicked_items] = -(1 << 10)
        rating2[i, clicked_items] = -(1 << 10)

        rating_K1, idx_K1 = torch.topk(rating1[i], k=topk)
        rating_K2, idx_K2 = torch.topk(rating2[i], k=topk)

        rating_K1, idx_K1 = rating_K1.cpu(), idx_K1.cpu()
        rating_K2, idx_K2 = rating_K2.cpu(), idx_K2.cpu()

        for idx in idx_K1:
            if idx.item() in test_groundTruth_items and idx.item() not in idx_K2:
                deg1.append(degree[idx + n_users].item())
                deg2.append(degree[i])

        if i > 1000:
            break

    plt.figure()
    plt.scatter(deg1, deg2)
    plt.xlabel('item degree')
    plt.ylabel('user degree')
    plt.savefig('degree.pdf')
    plt.close()


def plot_time(dataset, model):
    with open('result_' + model + '_' + dataset + '_neg0.txt', 'r') as f:
        data = list(f.readlines())[5]

    time, epoch, recall, ndcg = [], [], [], []
    for i in range(len(data)):
        if not i % 2:
            time.append(int(data[i][2:9]))
            epoch.append(int(data[i][14:15]))
            recall.append(int(data[i][58:64]))
            ndcg.append(int(data[i][104:110]))

    plt.figure()
    plt.plot(epoch, recall)
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.savefig('result/' + dataset + '/' + model +
                '/performance_curve.pdf', dpi=100)
    plt.close()
