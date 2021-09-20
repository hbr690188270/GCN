import numpy as np
import torch
from self_gcn import GCN, GCNLayer
import torch.nn.functional as F
from data_util import preprocess_adj, preprocess_features

from data_util import load_data, preprocess_features

import time

adj, features, orig_y_train, orig_y_val, orig_y_test, train_mask, val_mask, test_mask = load_data()

# device = torch.device("cuda")
device = torch.device("cpu")

dense_feat = features.todense()
dense_adj = adj.todense()

num_nodes, num_features, num_classes = dense_feat.shape[0], dense_feat.shape[1], orig_y_train.shape[1]

train_mask = torch.tensor(train_mask).to(device)
valid_mask = torch.tensor(val_mask).to(device)
test_mask = torch.tensor(test_mask).to(device)

feature_mat = torch.FloatTensor(dense_feat).to(device)
feature_mat = feature_mat / feature_mat.sum(1, keepdim = True).clamp_(min=1.)
adj_mat = torch.FloatTensor(dense_adj).to(device)
num_edges = torch.sum(adj_mat) / 2
y_train = torch.tensor(orig_y_train).to(device)
y_train = torch.argmax(y_train, dim = 1)[train_mask]

y_valid = torch.tensor(orig_y_val).to(device)
y_valid = torch.argmax(y_valid, dim = 1)[valid_mask]

y_test = torch.tensor(orig_y_test).to(device)
y_test = torch.argmax(y_test, dim = 1)[test_mask]

print("%d train, %d valid, %d test"%(len(y_train), len(y_valid), len(y_test)))

model = GCN(input_dim = num_features, hidden_dim = 16, output_dim = num_classes).to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01) 

def test(feature_mat, adj_mat):
    model.eval()
    logits, prob = model(feature_mat, adj_mat, attack = False)
    accs = []
    for mask in ["train", "valid", "test"]:
        pred = torch.argmax(prob[eval(mask + "_mask")], dim = 1)
        acc = pred.eq(eval("y_" + mask)).sum().item() / eval("y_" + mask).size(0)
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0

adj_mat = adj_mat + torch.eye(adj_mat.size(0)).to(device)
row_sum = torch.sum(adj_mat, dim = 1)
d_sqrt_inv = row_sum.pow_(-0.5).view(-1,1)
normalized_adj = torch.multiply(torch.multiply(d_sqrt_inv, adj_mat), d_sqrt_inv.view(1,-1))

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    logits, prob = model(feature_mat, normalized_adj, attack = False)
    prob = prob[train_mask]
    logits = logits[train_mask]

    loss = F.nll_loss(prob, y_train)
    # loss = F.cross_entropy(logits, y_train)
    loss.backward()
    optimizer.step()
    train_acc, val_acc, test_acc = test(feature_mat, normalized_adj)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
print("best acc: ", test_acc)

logits, prob = model(feature_mat, normalized_adj, attack = False)
label = torch.argmax(prob, dim = 1)



def bisection(a,eps,xi,ub=1):
    pa = torch.clip(a, 0, ub)
    if torch.sum(pa).item() <= eps:
        # print('np.sum(pa) <= eps !!!!')
        upper_S_update = pa
    else:
        mu_l = torch.min(a-1)
        mu_u = torch.max(a)
        #mu_a = (mu_u + mu_l)/2
        while torch.abs(mu_u - mu_l)>xi:
            #print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l)/2
            gu = torch.sum(torch.clip(a-mu_a, 0, ub)) - eps
            gu_l = torch.sum(torch.clip(a-mu_l, 0, ub)) - eps
            #print('gu:',gu)
            if gu == 0: 
                print('gu == 0 !!!!!')
                break
            if torch.sign(gu) == torch.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
            
        upper_S_update = torch.clip(a-mu_a, 0, ub)
        
    return upper_S_update

'''
attack begin
'''
perturb_ratio = 0.05
steps = 100
C = 200
lmd = 1
eps = num_edges * perturb_ratio
xi = 1e-5
attack_mask = train_mask + test_mask
attack_label = torch.argmax(torch.tensor(orig_y_train)[attack_mask], dim = 1).to(device).long()
model.eval()

upper_S_0 = torch.rand(dense_adj.shape, requires_grad = True).to(device)

A = torch.FloatTensor(dense_adj).to(device)
A.requires_grad = True
C_mat = 1 - 2 * A - torch.eye(A.size(0)).to(device)
mask = torch.Tensor(np.triu(np.ones(shape = A.size(),  dtype = np.float32), 1,)).to(device)
upper_S_real = torch.triu(upper_S_0, diagonal = 1) 

for epoch in range(steps):
    t = time.time()
    mu = C / np.sqrt(epoch + 1)


    upper_S_real2 = upper_S_real + torch.transpose(upper_S_real, 0, 1)
    modified_A = A + torch.multiply(upper_S_real2, C_mat)
    
    hat_A = modified_A + torch.eye(A.size(0)).to(device)
    row_sum = torch.sum(hat_A, dim = 1)
    d_sqrt = torch.sqrt(row_sum)
    d_sqrt_inv = (1 / d_sqrt).view(-1,1)
    support_real = torch.multiply(torch.multiply(d_sqrt_inv, hat_A), d_sqrt_inv.view(1,-1))
        
    logits, prob = model(feature_mat, support_real)

    test_pred = torch.argmax(prob, dim = 1)[test_mask]
    test_acc = test_pred.eq(y_test).sum().item()/y_test.size(0)
    test_loss = F.cross_entropy(prob[test_mask], y_test)
    print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(test_loss),
          "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))



    loss = F.cross_entropy(prob[attack_mask], attack_label)
    # loss.backward()
    Sgrad = torch.autograd.grad(outputs = loss, inputs = upper_S_real)[0]

    update = mu * Sgrad * lmd * mask

    a = upper_S_real + mu * Sgrad * lmd * mask
    upper_S_update = bisection(a, eps, xi)
    upper_S_real = upper_S_update

    if epoch == steps - 1:
        acc_record, support_record, p_ratio_record = [], [], []
        for i in range(10):
            print('random start!')
            randm = torch.tensor(np.random.uniform(size=(num_nodes, num_nodes))).to(device)
            upper_S_update = torch.where(upper_S_update > randm, 1, 0)
            upper_S_real = upper_S_update

            upper_S_real2 = upper_S_real + torch.transpose(upper_S_real, 0, 1)
            modified_A = A + torch.multiply(upper_S_real2, C_mat)
    
            hat_A = modified_A + torch.eye(A.size(0)).to(device)
            row_sum = torch.sum(hat_A, dim = 1)
            d_sqrt = torch.sqrt(row_sum)
            d_sqrt_inv = (1 / d_sqrt).view(-1,1)
            support_real = torch.multiply(torch.multiply(d_sqrt_inv, hat_A), d_sqrt_inv.view(1,-1))
            logits, prob = model(feature_mat, support_real)
            pred = torch.argmax(prob[test_mask], dim = 1)
            loss = F.cross_entropy(prob[test_mask], y_test)
            acc = pred.eq(y_test).sum().item() / y_test.size(0)


            pr = torch.count_nonzero(upper_S_update) / num_edges
            if pr <= perturb_ratio:
                acc_record.append(acc)
                support_record.append(support_real)
                p_ratio_record.append(pr)
            print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(loss),
                  "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
            print("perturb ratio", pr)
            print('random end!')
    

