import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from torch.utils.data import Dataset as BaseDataset
import os
from torch_geometric.utils import add_remaining_self_loops, degree


def read_cf_list(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]


def read_cf_by_user(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def process(train_data, test_data):
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    train_user_set, test_user_set, train_item_set = defaultdict(
        list), defaultdict(list), defaultdict(list)

    train_data[:, 1] += n_users
    test_data[:, 1] += n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
        train_item_set[int(i_id)].append(int(u_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))

    return n_users, n_items, train_user_set, test_user_set, train_item_set


def process_adj(data_cf, n_users, n_items):
    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1]  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    return torch.LongTensor(cf_).t()


def _bi_norm_lap(adj):
    # D^{-1/2}AD^{-1/2}
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()


def _si_norm_lap(adj):
    # D^{-1}A
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()


def load_data(args):
    print('reading train and test user-item set ...')
    train_cf = read_cf_list(os.getcwd() + '/data/' +
                            args.dataset + '/train' + str(args.split) + '.txt')
    test_cf = read_cf_list(os.getcwd() + '/data/' +
                           args.dataset + '/test' + str(args.split) + '.txt')

    n_users, n_items, train_user_set, test_user_set, train_item_set = process(
        train_cf, test_cf)

    print('building the adj mat ...')
    adj = process_adj(train_cf, n_users, n_items)

    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
        'train_item_set': train_item_set,
    }

    clicked_set = defaultdict(list)
    for key in user_dict:
        for user in user_dict[key]:
            clicked_set[user].extend(user_dict[key][user])

    print('loading over ...')
    return train_cf, test_cf, user_dict, n_users, n_items, clicked_set, adj


class Dataset(BaseDataset):
    def __init__(self, users, pos_items, neg_items, args, link_ratios=None):
        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items
        # self.link_ratios = link_ratios
        self.args = args

    def _get_feed_dict(self, index):

        # print(self.weight.shape)
        feed_dict = {
            'users': self.users[index],
            'pos_items': self.pos_items[index],
            'neg_items': self.neg_items[index],
            # 'link_ratios': self.link_ratios[index]
        }

        return feed_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        # feed_dicts: [dict1, dict2, ...]

        feed_dict = dict()

        feed_dict['users'] = torch.LongTensor([d['users'] for d in feed_dicts])
        feed_dict['pos_items'] = torch.LongTensor(
            [d['pos_items'] for d in feed_dicts])

        feed_dict['neg_items'] = torch.LongTensor(
            np.stack([d['neg_items'] for d in feed_dicts]))
        # feed_dict['link_ratios'] = torch.LongTensor(
        #     np.stack([d['link_ratios'] for d in feed_dicts]))

        feed_dict['idx'] = torch.cat(
            [feed_dict['users'], feed_dict['pos_items'], feed_dict['neg_items'].view(-1)])

        return feed_dict


def normalize_edge(edge_index, n_users, n_items):
    # edge_index, _ = add_remaining_self_loops(edge_index)

    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse.FloatTensor(edge_index, edge_weight, (n_users + n_items, n_users + n_items)), deg


def transform_data2(dataset):
    train_inter_mat, test_inter_mat = list(), list()

    # training
    lines = open(os.getcwd() + '/data/' + dataset +
                 '/train1.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip().split(",")
        train_inter_mat.append([int(tmps[0]), int(tmps[1])])

    lines = open(os.getcwd() + '/data/' + dataset +
                 '/test1.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip().split(",")
        test_inter_mat.append([int(tmps[0]), int(tmps[1])])

    np.savetxt(os.getcwd() + '/data/' + dataset +
               '/train1.txt', np.array(train_inter_mat), fmt='%i')
    np.savetxt(os.getcwd() + '/data/' + dataset +
               '/test1.txt', np.array(test_inter_mat), fmt='%i')


def transform_data1(dataset):
    train_inter_mat, test_inter_mat = list(), list()

    # training
    lines = open(os.getcwd() + '/data/' + dataset +
                 '/train.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]

        pos_ids = list(set(pos_ids))

        for i_id in pos_ids:
            train_inter_mat.append([u_id, i_id])

    # testing
    lines = open(os.getcwd() + '/data/' + dataset +
                 '/test.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]

        pos_ids = list(set(pos_ids))

        for i_id in pos_ids:
            test_inter_mat.append([u_id, i_id])

    np.savetxt(os.getcwd() + '/data/' + dataset +
               '/train1.txt', np.array(train_inter_mat), fmt='%i')
    np.savetxt(os.getcwd() + '/data/' + dataset +
               '/test1.txt', np.array(test_inter_mat), fmt='%i')


def preprocess_edges(file, dataset):
    edge_list = np.loadtxt(file, dtype=int)

    deg = degree(torch.tensor(edge_list[:, 0]),
                 num_nodes=max(edge_list[:, 0]) + 1)

    num_user, num_item = max(edge_list[:, 0]) + 1, max(edge_list[:, 1]) + 1
    adj = torch.zeros((num_user, num_item))
    adj[edge_list[:, 0], edge_list[:, 1]] = 1

    adj = adj[deg > 10]

    edge_list = adj.nonzero().numpy()
    idx = np.arange(len(edge_list))
    train_idx = np.random.choice(idx, size=int(
        edge_list.shape[0] * 0.8), replace=False)

    filtering = np.full(edge_list.shape[0], False, dtype=bool)
    filtering[train_idx] = True

    train_edge_list = edge_list[filtering]
    test_edge_list = edge_list[~filtering]

    np.savetxt(os.getcwd() + '/data/' + dataset +
               '/train1.txt', train_edge_list, fmt='%i')
    np.savetxt(os.getcwd() + '/data/' + dataset +
               '/test1.txt', test_edge_list, fmt='%i')


# preprocess_edges('./data/worldnews/worldnews_edges_2year.txt', 'worldnews')
