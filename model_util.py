import numpy as np
import torch
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros



class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops = True, normalize = True, bias = True,
                    **kwargs):
        kwargs['aggr'] = 'add'
        super(GCNLayer, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.weight = torch.nn.Linear(in_channels, out_channels, bias = False)
        # self.weight = 
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_paramters()

    def reset_paramters(self):
        # self.weight.reset_parameters()
        torch.nn.init.xavier_uniform_(self.weight.weight)
        zeros(self.bias)

    def add_remaining_self_loop(self, edge_index, num_nodes):
        '''
        edge_index: 2 * num_edges
        '''
        from_nodes, to_nodes = edge_index
        eq_nodes = (from_nodes != to_nodes)
        loop_index = torch.arange(0,num_nodes, dtype = from_nodes.dtype, device = from_nodes.device)
        loop_index = loop_index.unsqueeze(0).repeat(2,1)
        edge_index = torch.cat([edge_index[:, eq_nodes], loop_index], dim = 1)

        return edge_index

    def forward(self, x, edge_index,):
        ''''
        x: tensor,  num_nodes * hidden dim
        edge_index: tensor,   2 * num_edges
        '''
        num_nodes = x.size(0)
        edge_index = self.add_remaining_self_loop(edge_index, num_nodes)
        from_nodes, to_nodes = edge_index
        edge_weight = torch.ones(edge_index.size(1), device = from_nodes.device)
        degree_array = torch.zeros(size = [num_nodes,], device = from_nodes.device)
        degree_array.scatter_add_(dim = 0, index = to_nodes, src = edge_weight)
        deg_inv_sqrt = degree_array.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[from_nodes] * edge_weight * deg_inv_sqrt[to_nodes]
        x = self.weight(x)
        out = self.propagate(edge_index= edge_index, x = x, edge_weight = edge_weight, size = None)
        if self.bias is not None:
            out += self.bias
        return out


    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1,1)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
    

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_labels):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(num_features, 16)
        self.conv2 = GCNLayer(16, num_labels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training = self.training)
        logits = self.conv2(x, edge_index)
        prob = F.log_softmax(logits, dim = 1)
        return prob


