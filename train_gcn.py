import numpy as np
import torch
from self_gcn import GCN, GCNLayer
import torch.nn.functional as F

import scipy.sparse as sp


from data_util import load_data, preprocess_features

dataset = 'cora'

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

model = GCN(input_dim = num_features, hidden_dim = 16, output_dim = num_classes).to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01) 


def test(model, feature_mat, adj_mat):
    model.eval()
    logits, prob = model(feature_mat, adj_mat)
    accs = []
    for mask in ["train", "valid", "test"]:
        pred = torch.argmax(prob[eval(mask + "_mask")], dim = 1)
        acc = pred.eq(eval("y_" + mask)).sum().item() / eval("y_" + mask).size(0)
        accs.append(acc)
    return accs


best_val_acc = best_test_acc = 0

adj_mat = adj_mat + torch.eye(adj_mat.size(0)).to(device)
row_sum = torch.sum(adj_mat, dim = 1)
d_sqrt_inv = row_sum.pow_(-0.5).view(-1)
d_sqrt_inv[torch.isinf(d_sqrt_inv)] = 0
normalized_adj = torch.multiply(torch.multiply(d_sqrt_inv, adj_mat), d_sqrt_inv.view(1,-1))

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    logits, prob = model(feature_mat, normalized_adj)
    prob = prob[train_mask]
    logits = logits[train_mask]

    loss = F.nll_loss(prob, y_train)
    # loss = F.cross_entropy(logits, y_train)
    loss.backward()
    optimizer.step()
    train_acc, val_acc, test_acc = test(model, feature_mat, normalized_adj)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        torch.save(model, './models/cora/model.pt')
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, best_test_acc))
print("best acc: ", best_test_acc)

best_model = torch.load('./models/cora/model.pt')
best_model.eval()

logits, prob = best_model(feature_mat, normalized_adj)
pred_labels = torch.argmax(logits, dim = 1).detach().cpu().numpy()
tmp = np.zeros_like(orig_y_train)
tmp[np.arange(len(pred_labels)), pred_labels] = 1
tmp = orig_y_train + tmp * (1-np.expand_dims(train_mask.detach().cpu().numpy(),1))
np.save('label_'+ dataset + '.npy',tmp)
print('predicted label saved at '+'label_'+ dataset + '.npy')


