import os

from parse import parse_args
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import time

import numpy as np
import copy
import pickle

from utils import *
from evaluation import *
from model import *
from dataprocess import *


def run(model, optimizer, train_cf, clicked_set, user_dict, args):
    # data prepare for the model
    if args.model == 'MF':
        pass

    elif args.model in ['LightGCN', 'NGCF']:
        print('building the adj mat ...')
        adj = process_adj(train_cf, args.n_users, args.n_items)
        adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
        edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

        model.adj_sp_norm = adj_sp_norm.to(args.device)
        model.edge_index = edge_index.to(args.device)
        model.edge_weight = edge_weight.to(args.device)
        model.deg = deg.to(args.device)

    elif args.model == 'CAGCN':
        print('building the adj mat ...')
        adj = process_adj(train_cf, args.n_users, args.n_items)
        adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
        edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

        model.adj_sp_norm = adj_sp_norm.to(args.device)
        model.edge_index = edge_index.to(args.device)
        model.edge_weight = edge_weight.to(args.device)
        model.deg = deg.to(args.device)

        row, col = edge_index

        if args.type == 'jc':
            if args.dataset == 'amazon_book':
                cal_trend = co_ratio_deg_user_jacard_sp
            else:
                cal_trend = co_ratio_deg_user_jacard
        elif args.type == 'co':
            if args.dataset == 'amazon_book':
                cal_trend = co_ratio_deg_user_common_sp
            else:
                cal_trend = co_ratio_deg_user_common
        elif args.type == 'lhn':
            if args.dataset == 'amazon_book':
                cal_trend = co_ratio_deg_user_lhn_sp
            else:
                cal_trend = co_ratio_deg_user_lhn
        elif args.type == 'sc':
            if args.dataset == 'amazon_book':
                cal_trend = co_ratio_deg_user_sc_sp
            else:
                cal_trend = co_ratio_deg_user_sc

        path = os.getcwd() + '/data/' + args.dataset + \
            '/co_ratio_edge_weight_' + args.type + '.pt'

        if os.path.exists(path):
            trend = torch.load(path)
        else:
            print(args.dataset, 'calculate_CIR', 'count_time...')
            start = time.time()
            trend = cal_trend(
                adj_sp_norm, edge_index, deg, args)
            print('Preprocession', time.time() - start)

        norm_now = scatter_add(
            trend, col, dim=0, dim_size=args.n_users + args.n_items)[col]

        trend = args.trend_coeff * trend / norm_now + edge_weight

        model.trend = (trend).to(args.device)
        args.model = args.model + '-fusion-' + args.type

    best_val_recall, early_stop_count = -float('inf'), 0
    start = time.time()

    dataset = Dataset(
        users=train_cf[:, 0], pos_items=train_cf[:, 1], clicked_set=clicked_set, args=args)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=dataset.collate_batch, pin_memory=args.pin_memory)  # organzie the dataloader based on re-sampled negative pairs

    for epoch in range(args.epochs):
        """training"""
        model.train()
        for i, batch in enumerate(dataloader):
            batch = batch_to_gpu(batch, args.device)

            user_embs, pos_item_embs, neg_item_embs, user_embs0, pos_item_embs0, neg_item_embs0 = model(
                batch)

            bpr_loss = cal_bpr_loss(
                batch['users'], user_embs, pos_item_embs, neg_item_embs)

            # l2 regularization
            l2_loss = cal_l2_loss(
                user_embs0, pos_item_embs0, neg_item_embs0, user_embs0.shape[0])

            batch_loss = bpr_loss + args.l2 * l2_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        #******************evaluation****************
        if not epoch % 5:
            model.eval()
            res = PrettyTable()
            res.field_names = ["Time", "Epoch", "Recall",
                               "NDCG", "Precision", "Hit_ratio", "F1"]

            user_embs, item_embs = model.generate()
            res_val = eval_val(
                user_embs, item_embs, user_dict['train_user_set'], user_dict['val_user_set'], args)

            res.add_row(
                [format(time.time() - start, '.4f'), epoch, res_val['Recall'], res_val['NDCG'],
                 res_val['Precision'], res_val['Hit_ratio'], res_val['F1']])

            print(res)
            early_stop_count += 1

            if res_val['Recall'][3] > best_val_recall:
                best_val_recall = res_val['Recall'][3]

                early_stop_count = 0

                if args.save:
                    torch.save(model.state_dict(), os.getcwd() + '/trained_model/' + args.dataset +
                               '/' + args.model_name + '_' + str(args.trend_coeff) + '.pkl')

            if epoch > args.epochs // 2 and early_stop_count >= args.early_stop:
                break


def run_prediction(model, train_cf, clicked_set, user_dict, args):
    # data prepare for the model
    if args.model == 'MF':
        model.load_state_dict(torch.load(os.getcwd() + '/trained_model/' + args.dataset +
                                         '/' + args.model_name + '.pkl'))

    elif args.model in ['LightGCN', 'NGCF']:
        model.load_state_dict(torch.load(os.getcwd() + '/trained_model/' + args.dataset +
                                         '/' + args.model_name + '.pkl'))

        print('building the adj mat ...')
        adj = process_adj(train_cf, args.n_users, args.n_items)
        adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
        edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

        model.adj_sp_norm = adj_sp_norm.to(args.device)
        model.edge_index = edge_index.to(args.device)
        model.edge_weight = edge_weight.to(args.device)
        model.deg = deg.to(args.device)

    elif args.model == 'CAGCN':
        model.load_state_dict(torch.load(os.getcwd() + '/trained_model/' + args.dataset +
                                         '/' + args.model_name + '_' + str(args.trend_coeff) + '.pkl'))

        print('building the adj mat ...')
        adj = process_adj(train_cf, args.n_users, args.n_items)
        adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
        edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

        model.adj_sp_norm = adj_sp_norm.to(args.device)
        model.edge_index = edge_index.to(args.device)
        model.edge_weight = edge_weight.to(args.device)
        model.deg = deg.to(args.device)

        row, col = edge_index

        if args.type == 'jc':
            if args.dataset == 'amazon':
                cal_trend = co_ratio_deg_user_jacard_sp
            else:
                cal_trend = co_ratio_deg_user_jacard
        elif args.type == 'co':
            if args.dataset == 'amazon':
                cal_trend = co_ratio_deg_user_common_sp
            else:
                cal_trend = co_ratio_deg_user_common
        elif args.type == 'lhn':
            if args.dataset == 'amazon':
                cal_trend = co_ratio_deg_user_lhn_sp
            else:
                cal_trend = co_ratio_deg_user_lhn
        elif args.type == 'sc':
            if args.dataset == 'amazon':
                cal_trend = co_ratio_deg_user_sc_sp
            else:
                cal_trend = co_ratio_deg_user_sc

        path = os.getcwd() + '/data/' + args.dataset + \
            '/co_ratio_edge_weight_' + args.type + '.pt'

        if os.path.exists(path):
            trend = torch.load(path)
        else:
            print(args.dataset, 'calculate_CIR', 'count_time...')
            start = time.time()
            trend = cal_trend(
                adj_sp_norm, edge_index, deg, args)
            print('Preprocession', time.time() - start)

        norm_now = scatter_add(
            trend, col, dim=0, dim_size=args.n_users + args.n_items)[col]

        trend = args.trend_coeff * trend / norm_now + edge_weight

        model.trend = (trend).to(args.device)
        args.model = args.model + '-fusion-' + args.type

    user_embs, item_embs = model.generate()

    print(eval_val(user_embs, item_embs,
                   user_dict['train_user_set'], user_dict['val_user_set'], args))

    print(eval_test(
        user_embs, item_embs, user_dict['train_user_set'], user_dict['val_user_set'], user_dict['test_user_set'], args))


if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)

    """build dataset"""
    train_cf, val_cf, test_cf, user_dict, item_dict, args.n_users, args.n_items, args.deg_user = load_data(
        args)

    print('# of users:', args.n_users)
    print('# of items:', args.n_items)
    print('# of edges:', train_cf.shape[0] +
          val_cf.shape[0] + test_cf.shape[0])
    print('# of training edges:', train_cf.shape[0])
    print('# of validation edges:', val_cf.shape[0])
    print('# of testing edges:', test_cf.shape[0])
    print('Network density:',
          (train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users))

    clicked_set = user_dict['train_val_user_set']
    args.user_dict = user_dict
    args.item_dict = item_dict

    """build model"""
    if args.model == 'LightGCN':
        model = LightGCN(args).to(args.device)
    elif args.model == 'NGCF':
        model = NGCF(args).to(args.device)
    elif args.model == 'MF':
        model = MF(args).to(args.device)
    elif args.model == 'CAGCN':
        model = CAGCN(args).to(args.device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run(model, optimizer, train_cf, clicked_set, user_dict, args)

    run_prediction(model, train_cf, clicked_set, user_dict, args)
