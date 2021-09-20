import numpy as np
import torch
from self_gcn import GCN, GCNLayer
import torch.nn.functional as F
from data_util import preprocess_adj, preprocess_features

import scipy.sparse as sp


from data_util import load_data, preprocess_features
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()

device = torch.device("cpu")

dense_feat = features.todense()
dense_adj = adj.todense()

support = preprocess_adj(adj) 
# for non sparse
support = torch.tensor(sp.coo_matrix((support[1],(support[0][:,0],support[0][:,1])),shape=support[2]).toarray()).to(device)

# dense_adj = support



num_nodes, num_features, num_classes = dense_feat.shape[0], dense_feat.shape[1], y_train.shape[1]

train_mask = torch.tensor(train_mask).to(device)
valid_mask = torch.tensor(val_mask).to(device)
test_mask = torch.tensor(test_mask).to(device)

feature_mat = torch.FloatTensor(dense_feat).to(device)
# feature_mat = feature_mat / feature_mat.sum(1, keepdim = True).clamp_(min=1.)
feature_mat = feature_mat / feature_mat.sum(1, keepdim = True)

adj_mat = torch.FloatTensor(dense_adj).to(device)
y_train = torch.tensor(y_train).to(device)
y_train = torch.argmax(y_train, dim = 1)[train_mask]

y_valid = torch.tensor(y_val).to(device)
y_valid = torch.argmax(y_valid, dim = 1)[valid_mask]

y_test = torch.tensor(y_test).to(device)
y_test = torch.argmax(y_test, dim = 1)[test_mask]

print("%d train, %d valid, %d test"%(len(y_train), len(y_valid), len(y_test)))


adj_mat = adj_mat + torch.eye(adj_mat.size(0)).to(device)
row_sum = torch.sum(adj_mat, dim = 1)
d_sqrt_inv = row_sum.pow_(-0.5).view(-1)

d_sqrt_inv[torch.isinf(d_sqrt_inv)] = 0
d_mat_inv_sqrt = torch.diag(d_sqrt_inv)
normalized_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_mat),d_mat_inv_sqrt)
# normalized_adj = torch.multiply(torch.multiply(d_sqrt_inv, adj_mat), d_sqrt_inv.view(1,-1))

# normalized_adj = adj_mat
print(torch.sum(normalized_adj, dim = 0))
print(torch.sum(normalized_adj, dim = 1))


print(torch.sum(support, dim = 0))
print(torch.sum(support, dim = 1))

print(torch.sum(normalized_adj - support))
