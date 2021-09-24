import torch
import pickle

with open("./results/cora/result.pkl",'rb') as f:
    (robust_acc, robust_loss, clean_acc, clean_loss) = pickle.load(f)

transformed_robust_loss = []
transformed_clean_loss = []

for item in robust_loss:
    transformed_robust_loss.append(item.item())

for item in clean_loss:
    transformed_clean_loss.append(item.item())

with open("./results/cora/result2.pkl",'wb') as f:
    pickle.dump((robust_acc, transformed_robust_loss, clean_acc, transformed_clean_loss), f)
