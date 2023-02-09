import torch.nn.functional as F
import torch
from utils import *


def train(model, data, optimizer, args):
    model.train()
    optimizer.zero_grad()

    embed = model(data.x, data.edge_index, data.edge_weight)
    logits = F.log_softmax(embed, dim=1)

    loss = {}
    loss['train'] = F.nll_loss(
        logits[data.train_mask], data.y[data.train_mask])
    loss['val'] = F.nll_loss(logits[data.val_mask], data.y[data.val_mask])

    loss['train'].backward()
    optimizer.step()

    return loss


def evaluate(model, data, args):
    model.eval()

    with torch.no_grad():
        embed = model(data.x, data.edge_index,
                      data.edge_weight)

        logits = F.log_softmax(embed, dim=1)

    evals = {}

    pred_val = logits[data.val_mask].max(1)[1]
    evals['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()

    pred_test = logits[data.test_mask].max(1)[1]
    evals['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    return evals
