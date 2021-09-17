import numpy as np
from numpy.core.defchararray import upper 
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias = True):
        super(GCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = torch.nn.Linear(in_channels, out_channels, bias = False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_paramters()

    def reset_paramters(self):
        self.weight.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, inputs, adj):
        output = torch.matmul(adj, self.weight(inputs))
        output = output + self.bias
        return output

    
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.build_model()
    
    def build_model(self):
        self.conv1 = GCNLayer(self.input_dim, self.hidden_dim)
        self.conv2 = GCNLayer(self.hidden_dim, self.output_dim)
    
    def forward(self, input_x, adj,attack = True):
        if attack:
            mu = torch.IntTensor(1)
            A = adj
            mask = torch.Tensor(np.triu(np.ones(shape = A.size(),  dtype = np.float32), 1,))
            C = 1 - 2 * A - torch.eye(A.size(0))
            upper_S_0 = torch.zeros(A.size())
            upper_S_real = torch.triu(upper_S_0, diagonal = 1) 
            upper_S_real2 = upper_S_real + torch.transpose(upper_S_0, 0, 1)
            modified_A = A + torch.multiply(upper_S_real2, C)
            hat_A = modified_A + torch.eye(A.size(0))
            row_sum = torch.sum(hat_A, dim = 1)
            d_sqrt = torch.sqrt(row_sum)
            d_sqrt_inv = torch.linalg.inv(d_sqrt).view(-1,1)
            support_real = torch.multiply(torch.multiply(d_sqrt_inv, hat_A), d_sqrt_inv.view(1,-1))
            x = F.relu(self.conv1(input_x, support_real))
            x = F.dropout(x, training = self.training)
            logits = self.conv2(x, support_real)
            prob = F.log_softmax(logits, dim = 1)
            return logits, prob
        else:
            # x = F.dropout(input_x, training = self.training, p = 0.5)
            x = input_x
            x = F.relu(self.conv1(x, adj))
            x = F.dropout(x, training = self.training, p = 0.5)
            logits = self.conv2(x, adj)
            prob = F.log_softmax(logits, dim = 1)

            return logits, prob


