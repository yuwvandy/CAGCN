import os.path as osp
from torch_geometric.datasets import Planetoid, WebKB, Actor, Reddit, CoraFull, Coauthor
import torch_geometric.transforms as T
import torch
import numpy as np
from torch_geometric.utils import add_remaining_self_loops, degree


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1

    return mask


def get_dataset(name, normalize_features=False, transform=None):
    if(name in ['Cora', 'Citeseer', 'Pubmed']):
        load = Planetoid
    elif(name in ['Wisconsin', 'Cornell', 'Texas']):
        load = WebKB
    elif(name in ['Actor']):
        load = Actor
    elif(name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag', 'ogbn-proteins']):
        load = PygNodePropPredDataset
    elif(name in ['Reddit']):
        load = Reddit
    elif(name in ['CoraFull']):
        load = CoraFull
    elif(name in ['CS', 'Physics']):
        load = Coauthor

    dataset = load_dataset(
        name, load=load, normalize_features=normalize_features)

    return dataset


def load_dataset(name, load, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = load(root=path, name=name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def pre_process(dataset, name):
    data = dataset[0]
    if(name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']):
        data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
        data.edge_index = torch.stack([torch.cat([data.edge_index[0], data.edge_index[1]]), torch.cat(
            [data.edge_index[1], data.edge_index[0]])])

        split_idx = dataset.get_idx_split()
        train_mask, val_mask, test_mask = index_to_mask(split_idx['train'], data.x.size(0)), index_to_mask(
            split_idx['valid'], data.x.size(0)), index_to_mask(split_idx['test'], data.x.size(0))

        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
        data.y = data.y.view(-1, )

    return data


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data
