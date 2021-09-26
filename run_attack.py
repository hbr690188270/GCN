import numpy as np
import torch
from self_gcn import GCN, GCNLayer
import torch.nn.functional as F
from data_util import preprocess_adj, preprocess_features

from data_util import load_data, preprocess_features

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
model = torch.load("./models/cora/rob_model_std.pt").to(device)
model.eval()
logits, prob = model(feature_mat, normalized_adj)
label = torch.argmax(prob, dim = 1)

test_loss = F.cross_entropy(logits[test_mask], y_test)
test_pred = label[test_mask]
test_acc = test_pred.eq(y_test).sum().item()/y_test.size(0)
print("test loss: {}, test acc {}".format(test_loss, test_acc))



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


def bisection2(a,eps,xi,ub=1):
    pa = np.clip(a, 0, ub)
    if np.sum(pa) <= eps:
        # print('np.sum(pa) <= eps !!!!')
        upper_S_update = pa
    else:
        mu_l = np.min(a-1)
        mu_u = np.max(a)
        #mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l)>xi:
            #print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l)/2
            gu = np.sum(np.clip(a-mu_a, 0, ub)) - eps
            gu_l = np.sum(np.clip(a-mu_l, 0, ub)) - eps
            #print('gu:',gu)
            if gu == 0: 
                print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
            
        upper_S_update = np.clip(a-mu_a, 0, ub)
        
    return upper_S_update

'''
attack begin
'''
perturb_ratio = 0.05
steps = 200
C = 200
lmd = 1
eps = num_edges * perturb_ratio
xi = 1e-5
# attack_mask = train_mask.detach() + test_mask.detach()
# attack_mask = test_mask.detach()[corr_test_mask]
attack_label = y_all[test_mask].to(device).long()

upper_S_0 = torch.zeros(dense_adj.shape, requires_grad = True, device = device, dtype = torch.float)
A = torch.tensor(dense_adj, dtype = torch.float, device = device, requires_grad=True)
C_mat = 1 - 2 * A - torch.eye(A.size(0), device = device, dtype = torch.float)
mask = torch.tensor(np.triu(np.ones(shape = A.size(),  dtype = np.float32), 1,), device = device)

for epoch in range(steps):
    t = time.time()
    mu = C / np.sqrt(epoch + 1)

    # mu = 1
    upper_S_real = torch.triu(upper_S_0, diagonal = 1)
    upper_S_real2 = upper_S_real + torch.transpose(upper_S_real, 1, 0)
    modified_A = A + torch.multiply(upper_S_real2, C_mat)
    
    hat_A = modified_A + torch.eye(A.size(0), device = device)
    row_sum = torch.sum(hat_A, dim = 1)
    d_sqrt_inv = row_sum.pow_(-0.5).view(-1,1)
    support_real = torch.multiply(torch.multiply(d_sqrt_inv, hat_A), d_sqrt_inv.view(1,-1))

    logits, prob = model(feature_mat, support_real)
    loss = F.cross_entropy(logits[test_mask], attack_label)


    test_pred = torch.argmax(prob, dim = 1)[test_mask]
    test_acc = test_pred.eq(y_test).sum().item()/y_test.size(0)
    test_loss = F.cross_entropy(logits[test_mask], y_test)
    print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(test_loss),
          "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))

    # test_pred = torch.argmax(prob, dim = 1)[attack_mask]
    # test_acc = test_pred.eq(attack_label).sum().item()/attack_label.size(0)
    # print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(loss),
    #       "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))


    upper_S_real.retain_grad()
    # A.retain_grad()
    loss.backward(retain_graph = True)
    # test_loss.backward(retain_graph = True)

    # Sgrad = torch.autograd.grad(outputs = loss, inputs = upper_S_real)[0]

    Sgrad = upper_S_real.grad
    a = upper_S_real + mu * Sgrad * lmd * mask
    res = bisection(a, eps, xi)
    upper_S_0.data = res
    # upper_S_real.data = res
    upper_S_0.grad.zero_()
    upper_S_real.grad.zero_()

    # Agrad = A.grad
    # A = mu * Agrad + A
    # A.grad.zero_()

    upper_S_update_tmp = res[:]
    if epoch == steps - 1:
        acc_record, support_record, p_ratio_record = [], [], []
        for i in range(10):
            print('random start!')
            randm = torch.tensor(np.random.uniform(size=(num_nodes, num_nodes))).to(device)
            upper_S_update = torch.where(upper_S_update_tmp > randm, 1, 0)
            upper_S_real = torch.triu(upper_S_update, diagonal = 1)

            upper_S_real2 = upper_S_real + torch.transpose(upper_S_real, 0, 1)
            modified_A = A + torch.multiply(upper_S_real2, C_mat)
    
            hat_A = modified_A + torch.eye(A.size(0)).to(device)
            row_sum = torch.sum(hat_A, dim = 1)
            d_sqrt = torch.sqrt(row_sum)
            d_sqrt_inv = (1 / d_sqrt).view(-1,1)
            support_real = torch.multiply(torch.multiply(d_sqrt_inv, hat_A), d_sqrt_inv.view(1,-1))
            logits, prob = model(feature_mat, support_real)
            pred = torch.argmax(prob[test_mask], dim = 1)
            loss = F.cross_entropy(prob[test_mask],y_test)
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
        print("acc list: ", acc_record)
        final_support = support_record[np.argmin(acc_record)]

