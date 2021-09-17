import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from model_util import GCN
import torch.nn.functional as F

dataset = 'Cora'
path = './dataset/cora/'
dataset = Planetoid(root = path, name = dataset, transform=T.NormalizeFeatures())
data = dataset[0]

num_features = dataset.num_features
num_labels = dataset.num_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = GCN(num_features, num_labels).to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.



def train(data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


print(data.y[data.train_mask].size(), data.y[data.val_mask].size(), data.y[data.test_mask].size())


best_val_acc = test_acc = 0
for epoch in range(300):
    train(data)
    train_acc, val_acc, tmp_test_acc = test(data)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

