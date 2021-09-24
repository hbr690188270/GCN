import numpy as np
import torch
from self_gcn import GCN, GCNLayer
import torch.nn.functional as F
from data_util import preprocess_adj, preprocess_features

from data_util import load_data, preprocess_features
from PGDAttack import pgd_attack
import time

adj, features, orig_y_train, orig_y_val, orig_y_test, all_orig_label, train_mask, val_mask, test_mask = load_data()

device = torch.device("cuda")
# device = torch.device("cpu")

dense_feat = features.todense()
dense_adj = adj.todense()

num_nodes, num_features, num_classes = dense_feat.shape[0], dense_feat.shape[1], orig_y_train.shape[1]

train_mask = torch.tensor(train_mask).to(device)
valid_mask = torch.tensor(val_mask).to(device)
test_mask = torch.tensor(test_mask).to(device)

feature_mat = torch.FloatTensor(dense_feat).to(device)
feature_mat = feature_mat / feature_mat.sum(1, keepdim = True).clamp_(min=1.)
adj_mat = torch.FloatTensor(dense_adj).to(device)
num_edges = (torch.sum(adj_mat) / 2).item()
y_train = torch.tensor(orig_y_train).to(device)
y_train = torch.argmax(y_train, dim = 1)[train_mask]

y_valid = torch.tensor(orig_y_val).to(device)
y_valid = torch.argmax(y_valid, dim = 1)[valid_mask]

y_test = torch.tensor(orig_y_test).to(device)
y_test = torch.argmax(y_test, dim = 1)[test_mask]

y_all = torch.tensor(all_orig_label).to(device)
y_all = torch.argmax(y_all, dim = 1)

print("%d train, %d valid, %d test, total %d "%(len(y_train), len(y_valid), len(y_test), len(y_all)))


adj_mat = adj_mat + torch.eye(adj_mat.size(0)).to(device)
row_sum = torch.sum(adj_mat, dim = 1)
d_sqrt_inv = row_sum.pow_(-0.5).view(-1,1)
normalized_adj = torch.multiply(torch.multiply(d_sqrt_inv, adj_mat), d_sqrt_inv.view(1,-1))

# model = GCN(input_dim = num_features, hidden_dim = 16, output_dim = num_classes).to(device)
model = torch.load("./models/cora/model.pt").to(device)
model.eval()
logits, prob = model(feature_mat, normalized_adj)
label = torch.argmax(prob, dim = 1)


orig_adj = torch.tensor(dense_adj, dtype = torch.float, device = device, requires_grad=True)
attacker = pgd_attack(model, features = feature_mat, orig_adj = orig_adj, ratio = 0.05, device = device)
attacker.perturb(y_test = y_test, test_mask = test_mask, k = 100)

