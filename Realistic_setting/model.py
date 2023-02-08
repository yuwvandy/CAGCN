import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_scatter import scatter
from utils import softmax


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, args):
        super(GraphConv, self).__init__()
        self.args = args

    def forward(self, embed, adj_sp_norm, edge_index, edge_weight, deg):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        agg_embed = embed
        embs = [embed]

        row, col = edge_index

        for hop in range(self.args.n_hops):
            out = agg_embed[row] * edge_weight.unsqueeze(-1)
            agg_embed = scatter(
                out, col, dim=0, dim_size=self.args.n_users + self.args.n_items, reduce='add')

            # method2: sparse matrix multiplication
            # agg_embed = torch.sparse.mm(adj_sp_norm, agg_embed)
            embs.append(agg_embed)

        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]

        return embs[:self.args.n_users, :], embs[self.args.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, args):
        super(LightGCN, self).__init__()

        self.args = args

        self._init_weight()
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_users + self.args.n_items, self.args.embedding_dim)))

    def _init_model(self):
        if self.args.model == 'LightGCN':
            return GraphConv(self.args)

    def batch_generate(self, user, pos_item, neg_item):
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_gcn_embs, item_gcn_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        user_embs = user_gcn_embs[user]
        pos_item_embs = item_gcn_embs[pos_item - self.args.n_users]
        neg_item_embs = item_gcn_embs[neg_item - self.args.n_users]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, self.embeds[user], self.embeds[pos_item], self.embeds[neg_item]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self):
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_embs, item_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        return user_embs, item_embs

    def generate_layers(self):
        return self.gcn(self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)


class NGCF(nn.Module):
    def __init__(self, args):
        super(NGCF, self).__init__()
        self.args = args

        self._init_weight()

    def _init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_users + self.args.n_items, self.args.embedding_dim)))

        self.weight_dict = nn.ParameterDict()
        layers = [self.args.embedding_dim] + self.args.layer_sizes
        for k in range(len(self.args.layer_sizes)):
            self.weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                                         layers[k + 1])))})
            self.weight_dict.update({'b_gc_%d' % k: nn.Parameter(
                initializer(torch.empty(1, layers[k + 1])))})

            self.weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                                         layers[k + 1])))})
            self.weight_dict.update({'b_bi_%d' % k: nn.Parameter(
                initializer(torch.empty(1, layers[k + 1])))})

    def forward(self, batch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        ego_embeddings = self.embeds
        all_embeddings = [ego_embeddings]

        for k in range(len(self.args.layer_sizes)):
            row, col = self.edge_index
            out = ego_embeddings[row] * self.edge_weight.unsqueeze(-1)
            side_embeddings = scatter(
                out, col, dim=0, dim_size=self.args.n_users + self.args.n_items, reduce='add')

            # side_embeddings = torch.sparse.mm(self.adj_sp_norm, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                sum_embeddings + bi_embeddings)

            # message dropout.
            # ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)

        user_embs, pos_item_embs, neg_item_embs = all_embeddings[user,
                                                                 :], all_embeddings[pos_item, :], all_embeddings[neg_item, :]

        return user_embs, pos_item_embs, neg_item_embs, user_embs, pos_item_embs, neg_item_embs

    def generate(self):
        ego_embeddings = self.embeds
        all_embeddings = [ego_embeddings]

        for k in range(len(self.args.layer_sizes)):
            side_embeddings = torch.sparse.mm(self.adj_sp_norm, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                sum_embeddings + bi_embeddings)

            # message dropout.
            # ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)

        return all_embeddings[:self.args.n_users], all_embeddings[self.args.n_users:]


class MF(nn.Module):
    def __init__(self, args):
        super(MF, self).__init__()
        self.args = args

        self._init_weight()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_users + self.args.n_items, self.args.embedding_dim)))

    def generate(self):
        user_embs, item_embs = self.embeds[:self.args.n_users,
                                           :], self.embeds[self.args.n_users:, :]

        return user_embs, item_embs

    def batch_generate(self, user, pos_item, neg_item):
        user_embs = self.embeds[user]
        pos_item_embs = self.embeds[pos_item]
        neg_item_embs = self.embeds[neg_item]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        batch_size = user.shape[0]

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, user_embs, pos_item_embs, neg_item_embs


class GraphConv_CA(nn.Module):
    """
    Collaborative Adaptive Graph Convolutional Network
    """

    def __init__(self, args):
        super(GraphConv_CA, self).__init__()
        self.args = args

    def forward(self, embed, adj_sp_norm, edge_index, edge_weight, trend):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        agg_embed = embed
        embs = [embed]

        row, col = edge_index

        for hop in range(self.args.n_hops):
            out = agg_embed[row] * \
                (trend).unsqueeze(-1)
            agg_embed = scatter(
                out, col, dim=0, dim_size=self.args.n_users + self.args.n_items, reduce='add')

            embs.append(agg_embed)

        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]

        return embs


class CAGCN(nn.Module):
    def __init__(self, args):
        super(CAGCN, self).__init__()

        self.args = args

        self._init_weight()
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_users + self.args.n_items, self.args.embedding_dim)))

    def _init_model(self):
        return GraphConv_CA(self.args)

    def batch_generate(self, user, pos_item, neg_item):
        embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.trend)

        embs = self.pooling(embs)

        user_gcn_embs, item_gcn_embs = embs[:
                                            self.args.n_users], embs[self.args.n_users:]

        user_embs = user_gcn_embs[user]
        pos_item_embs = item_gcn_embs[pos_item - self.args.n_users]
        neg_item_embs = item_gcn_embs[neg_item - self.args.n_users]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, self.embeds[user], self.embeds[pos_item], self.embeds[neg_item]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self):
        embs = self.gcn(self.embeds, self.adj_sp_norm,
                        self.edge_index, self.edge_weight, self.trend)

        embs = self.pooling(embs)

        return embs[:self.args.n_users], embs[self.args.n_users:]

    def generate_layers(self):
        return self.gcn(self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.trend)
