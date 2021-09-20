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
        # x = F.dropout(input_x, training = self.training, p = 0.5)
        x = input_x
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, training = self.training, p = 0.5)
        logits = self.conv2(x, adj)
        prob = F.log_softmax(logits, dim = 1)

        return logits, prob


