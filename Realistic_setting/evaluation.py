import numpy as np
from collections import defaultdict
from utils import *
from torch_scatter import scatter

np.set_printoptions(precision=4)


def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')


def Hit_at_k(r, k):
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    # print(right_pred, 2213123213)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    precis = right_pred / precis_n
    return {'Recall': recall, 'Precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """

    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix

    # print(max_r[0], pred_data[0])
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)

    idcg[idcg == 0.] = 1.  # it is OK since when idcg == 0, dcg == 0
    ndcg = dcg / idcg
    # ndcg[np.isnan(ndcg)] = 0.

    return ndcg


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pre, recall, ndcg, hit_ratio, F1 = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs = NDCGatK_r(groundTrue, r, k)
        hit_ratios = Hit_at_k(r, k)

        hit_ratio.append(sum(hit_ratios))
        pre.append(sum(ret['Precision']))
        recall.append(sum(ret['Recall']))
        ndcg.append(sum(ndcgs))

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s = 2 * ret['Precision'] * ret['Recall'] / temp
        # F1s[np.isnan(F1s)] = 0

        F1.append(sum(F1s))

    return {'Recall': np.array(recall),
            'Precision': np.array(pre),
            'NDCG': np.array(ndcg),
            'F1': np.array(F1),
            'Hit_ratio': np.array(hit_ratio)}


def test_one_batch_group(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pres, recalls, ndcgs, hit_ratios, F1s = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs.append(NDCGatK_r(groundTrue, r, k))
        hit_ratios.append(Hit_at_k(r, k))
        recalls.append(ret['Recall'])
        pres.append(ret['Precision'])

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s.append(2 * ret['Precision'] * ret['Recall'] / temp)
        # F1s[np.isnan(F1s)] = 0

    return np.stack(recalls).transpose(1, 0), np.stack(ndcgs).transpose(1, 0), np.stack(hit_ratios).transpose(1, 0), np.stack(pres).transpose(1, 0), np.stack(F1s).transpose(1, 0)


def eval_train(user_embs, item_embs, train_user_set, args):
    results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks))}

    train_users = torch.tensor(list(train_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(train_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            groundTruth_items = [train_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, args.topks))

        for result in pre_results:
            results['Recall'] += result['Recall']
            results['Precision'] += result['Precision']
            results['NDCG'] += result['NDCG']
            results['F1'] += result['F1']
            results['Hit_ratio'] += result['Hit_ratio']

        results['Recall'] /= len(train_users)
        results['Precision'] /= len(train_users)
        results['NDCG'] /= len(train_users)
        results['F1'] /= len(train_users)
        results['Hit_ratio'] /= len(train_users)

    return results


def eval_val(user_embs, item_embs, train_user_set, val_user_set, args):
    results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks))}

    val_users = torch.tensor(list(val_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(val_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            clicked_items = [train_user_set[user.item()] - args.n_users
                             for user in batch_users]
            groundTruth_items = [val_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, args.topks))

        for result in pre_results:
            results['Recall'] += result['Recall']
            results['Precision'] += result['Precision']
            results['NDCG'] += result['NDCG']
            results['F1'] += result['F1']
            results['Hit_ratio'] += result['Hit_ratio']

        results['Recall'] /= len(val_users)
        results['Precision'] /= len(val_users)
        results['NDCG'] /= len(val_users)
        results['F1'] /= len(val_users)
        results['Hit_ratio'] /= len(val_users)

    return results


def eval_test(user_embs, item_embs, train_user_set, val_user_set, test_user_set, args):
    results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks))}

    test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            # consider further
            clicked_items = [np.concatenate((train_user_set[user.item()] - args.n_users, val_user_set[user.item()] - args.n_users))
                             for user in batch_users]

            groundTruth_items = [test_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, args.topks))

        for result in pre_results:
            results['Recall'] += result['Recall']
            results['Precision'] += result['Precision']
            results['NDCG'] += result['NDCG']
            results['F1'] += result['F1']
            results['Hit_ratio'] += result['Hit_ratio']

        results['Recall'] /= len(test_users)
        results['Precision'] /= len(test_users)
        results['NDCG'] /= len(test_users)
        results['F1'] /= len(test_users)
        results['Hit_ratio'] /= len(test_users)

    return results


def eval_test_group(user_embs, item_embs, train_user_set, val_user_set, test_user_set, deg, args):
    node_res = {'Precision': {},
                'Recall': {},
                'NDCG': {},
                'Hit_ratio': {},
                'F1': {}}

    res_group = {'Precision': {},
                 'Recall': {},
                 'NDCG': {},
                 'Hit_ratio': {},
                 'F1': {}}

    deg_keys = ['[0, 100)', '[100, 200)', '[200, 300)',
                '[300, 400)', '[400, 500)', '[500, Inf)']

    for key in node_res:
        for k in args.topks:
            node_res[key][k] = defaultdict(list)
            # key-degree group, val - (mean, std, num_nodes)
            res_group[key][k] = defaultdict(list)

    test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            # print(rating_batch)

            # consider further
            clicked_items = [np.concatenate((train_user_set[user.item()] - args.n_users, val_user_set[user.item()] - args.n_users))
                             for user in batch_users]

            groundTruth_items = [test_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()
            # rating_K = torch.tensor([np.random.choice(
            #     np.arange(rating_batch.shape[1]), max(args.topks)) for _ in range(rating_batch.shape[0])])

            users_list.append(batch_users.tolist())
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        for users, X in zip(users_list, zip(ratings_list, groundTruth_items_list)):
            recalls, ndcgs, hit_ratios, precisions, F1s = test_one_batch_group(
                X, args.topks)

            for i in range(len(recalls)):
                id_deg = int(deg[users[i]] // 100)
                if id_deg >= 5:
                    id_deg = 5

                for j, k in enumerate(args.topks):
                    node_res['Recall'][k][deg_keys[id_deg]].append(
                        recalls[i, j])
                    node_res['Precision'][k][deg_keys[id_deg]].append(
                        precisions[i, j])
                    node_res['NDCG'][k][deg_keys[id_deg]].append(ndcgs[i, j])
                    node_res['F1'][k][deg_keys[id_deg]].append(F1s[i, j])
                    node_res['Hit_ratio'][k][deg_keys[id_deg]].append(
                        hit_ratios[i, j])

        for key in node_res:
            for k in args.topks:
                for deg in deg_keys:
                    res_group[key][k][deg] = [np.mean(node_res[key][k][deg]), np.std(
                        node_res[key][k][deg]), len(node_res[key][k][deg])]

    return res_group


def eval_group_loss_dist(user_embs, item_embs, user_item_interaction_matrix, user_group_idx):
    #positive - 1, negative - (-1)
    user_item_dot_prod = torch.matmul(user_embs, item_embs.t())

    pos_mask = (user_item_interaction_matrix > 0) * 1.
    neg_mask = (user_item_interaction_matrix < 0) * 1.

    emb_mag_user = user_embs.norm(dim=1).view(-1, 1)
    norm_user_item_dot_prod = user_item_dot_prod / emb_mag_user

    user_simi = (norm_user_item_dot_prod * pos_mask).sum(dim=1) / pos_mask.sum(dim=1) - \
        (norm_user_item_dot_prod * neg_mask).sum(dim=1) / neg_mask.sum(dim=1)

    user_simi_group = scatter(user_simi, user_group_idx, reduce="mean")

    return user_simi_group


# def cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
#     pos_scores = torch.sum(
#         torch.mul(user_embs, pos_item_embs), axis=1)

#     neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(
#         dim=1), neg_item_embs), axis=-1)

#     bpr_loss = torch.mean(
#         torch.log(1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1)))

#     # bpr_loss = (link_ratios*torch.log(1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1))).mean()

#     return bpr_loss
