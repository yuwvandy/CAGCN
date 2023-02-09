from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
import torch.nn.functional as F
import torch
from utils import *
from torch_geometric.nn import GCNConv, ChebConv
from torch.nn.parameter import Parameter
import math



class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        self.args = args

        self.lin1 = Linear(args.num_features, args.hidden, bias=False)
        self.bias1 = Parameter(torch.FloatTensor(args.hidden))

        self.lin2 = Linear(args.hidden, args.num_classes, bias=False)
        self.bias2 = Parameter(torch.FloatTensor(args.num_classes))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        stdv = 0.9 / math.sqrt(self.args.hidden)
        self.bias1.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_weight):
        x = self.lin1(x)
        x = F.relu(propagate(x, edge_index, edge_weight) + self.bias1)
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.lin2(x)
        x = propagate(x, edge_index, edge_weight) + self.bias2

        return x
