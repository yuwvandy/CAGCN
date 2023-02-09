import argparse
from dataset import *
from learn import *
from model import *
from utils import *
from os import path
from tqdm import tqdm
import random
from torch import tensor
from torch_scatter import scatter_max, scatter_add


def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')

    acc = np.zeros(args.runs)

    data = data.to(args.device)
    model = GCN(args).to(args.device)

    for count in pbar:
        seed_everything(args.seed + count)

        model.reset_parameters()

        optimizer = torch.optim.Adam([
            dict(params=model.lin1.parameters(), weight_decay=args.wd1),
            dict(params=model.bias1, weight_decay=args.wd1),
            dict(params=model.lin2.parameters(), weight_decay=args.wd2),
            dict(params=model.bias2, weight_decay=args.wd2)], lr=args.lr)

        best_val_loss = float('inf')
        val_loss_history = []

        for epoch in range(0, args.epochs):
            loss = train(model, data, optimizer, args)

            evals = evaluate(model, data, args)

            if loss['val'] < best_val_loss:
                best_val_loss = loss['val']
                test_acc = evals['test']

                # torch.save(model.state_dict(), 'model.pkl')

            val_loss_history.append(loss['val'])
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                if loss['val'] > tmp.mean().item():
                    break

        acc[count] = test_acc
        print(test_acc)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd1', type=float, default=0.0008)
    parser.add_argument('--wd2', type=float, default=0.0000)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.9)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1028)
    parser.add_argument('--cir_coeff', type=float, default=0.8)
    parser.add_argument('--model', type=str, default='cagcn')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # import the dataset
    dataset = get_dataset(args.dataset, args.normalize_features)
    data = pre_process(dataset, args.dataset)
    args.num_features, args.num_classes = dataset.num_features, dataset.num_classes

    data.edge_index, _ = add_remaining_self_loops(
        data.edge_index, num_nodes=data.x.size(0))

    # calculate the degree normalize term
    row, col = data.edge_index
    deg = degree(col, data.x.size(0), dtype=data.x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    new_edge_weight = cal_sc(data.edge_index, data.x, args)

    norm = scatter_add(edge_weight, col, dim=0,
                       dim_size=data.x.size(0))[col]
    norm_now = scatter_add(new_edge_weight, col, dim=0,
                           dim_size=data.x.size(0))[col]

    new_edge_weight = args.cir_coeff * \
        new_edge_weight / norm_now + edge_weight


    data.edge_weight = new_edge_weight

    acc = run(data, args)

    print(np.mean(acc), np.std(acc))
