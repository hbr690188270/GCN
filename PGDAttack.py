import numpy as np 
import torch
import torch.nn.functional as F
import time
import logging
logger = logging.getLogger("adv_train")

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


class pgd_attack():
    def __init__(self, model, features, orig_adj, ratio, xi = 1e-5, device = torch.device('cuda')):
        self.victim_model = model
        self.features = features
        self.orig_adj = orig_adj
        self.ratio = ratio
        self.xi = xi
        self.device = device
        self.num_nodes = self.features.size(0)
        self.num_edges = (torch.sum(self.orig_adj) / 2).item()
    def perturb(self, y_test, test_mask, k = 100, eps=None, visualize = True):
        self.victim_model.eval()
        if eps:
            self.eps = eps
        else:
            self.eps = self.num_edges * self.ratio
        upper_S_0 = torch.zeros(self.orig_adj.shape, requires_grad = True, device = self.device, dtype = torch.float)
        A = torch.tensor(self.orig_adj.detach().cpu().numpy(), dtype = torch.float, device = self.device, requires_grad=True)
        C_mat = 1 - 2 * A - torch.eye(A.size(0), device = self.device, dtype = torch.float)
        mask = torch.tensor(np.triu(np.ones(shape = A.size(),  dtype = np.float32), 1,), device = self.device)

        C = 200
        for epoch in range(k):
            t = time.time()
            mu = C / np.sqrt(epoch + 1)
            # mu = 1
            upper_S_real = torch.triu(upper_S_0, diagonal = 1)
            upper_S_real2 = upper_S_real + torch.transpose(upper_S_real, 1, 0)
            modified_A = A + torch.multiply(upper_S_real2, C_mat)
            
            hat_A = modified_A + torch.eye(A.size(0), device = self.device)
            row_sum = torch.sum(hat_A, dim = 1)
            d_sqrt_inv = row_sum.pow_(-0.5).view(-1,1)
            support_real = torch.multiply(torch.multiply(d_sqrt_inv, hat_A), d_sqrt_inv.view(1,-1))

            logits, prob = self.victim_model(self.features, support_real)
            loss = F.cross_entropy(logits[test_mask], y_test)


            test_pred = torch.argmax(prob, dim = 1)[test_mask]
            test_acc = test_pred.eq(y_test).sum().item()/y_test.size(0)
            if visualize:
                print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(loss),
                    "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))
                logger.info("Step: {: 04d}, test_loss= {:.5f}, test_acc={:.5f}, time={:.5f}".format(epoch + 1,loss, test_acc, time.time() - t))

            upper_S_real.retain_grad()
            # A.retain_grad()
            loss.backward(retain_graph = True)

            # Sgrad = torch.autograd.grad(outputs = loss, inputs = upper_S_real)[0]
            
            Sgrad = upper_S_real.grad
            a = upper_S_real + mu * Sgrad * mask
            res = bisection(a, self.eps, self.xi)
            upper_S_0.data = res
            # upper_S_real.data = res
            upper_S_0.grad.zero_()
            upper_S_real.grad.zero_()


            upper_S_update_tmp = res[:]
            if epoch == k - 1:
                acc_record, support_record, p_ratio_record = [], [], []
                for _ in range(20):
                    if visualize:
                        print('random start!')
                        logger.info("randm start!")
                    randm = torch.tensor(np.random.uniform(size=(self.num_nodes, self.num_nodes))).to(self.device)
                    upper_S_update = torch.where(upper_S_update_tmp > randm, 1, 0)
                    upper_S_real = upper_S_update

                    upper_S_real2 = upper_S_real + torch.transpose(upper_S_real, 0, 1)
                    modified_A = A + torch.multiply(upper_S_real2, C_mat)
            
                    hat_A = modified_A + torch.eye(A.size(0)).to(self.device)
                    row_sum = torch.sum(hat_A, dim = 1)
                    d_sqrt = torch.sqrt(row_sum)
                    d_sqrt_inv = (1 / d_sqrt).view(-1,1)
                    support_real = torch.multiply(torch.multiply(d_sqrt_inv, hat_A), d_sqrt_inv.view(1,-1))
                    logits, prob = self.victim_model(self.features, support_real)
                    pred = torch.argmax(prob[test_mask], dim = 1)
                    loss = F.cross_entropy(prob[test_mask], y_test)
                    acc = pred.eq(y_test).sum().item() / y_test.size(0)


                    pr = torch.count_nonzero(upper_S_update) / self.num_edges
                    if pr <= self.ratio:
                        acc_record.append(acc)
                        support_record.append(support_real.detach().cpu().numpy())
                        p_ratio_record.append(pr.item())
                    if visualize:
                        print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(loss),
                            "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
                        print("perturb ratio", pr)
                        print('random end!')
                        logger.info("Step: {: 04d}, test_loss= {:.5f}, test_acc={:.5f}, time={:.5f}".format(epoch + 1,loss, acc, time.time() - t))
                        logger.info("perturb ratio: %f"%(pr))
                        logger.info("random end!")

                print("acc list: ", acc_record)
                logger.info("acc list {}".format(acc_record))
                if len(acc_record) == 0:
                    return ()
                return (support_record[np.argmin(acc_record)],)
            
 


