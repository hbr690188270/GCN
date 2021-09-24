import pickle
import matplotlib.pyplot as plt

with open("./results/cora/result.pkl",'rb') as f:
    (robust_acc, robust_loss, clean_acc, clean_loss) = pickle.load(f)
epoch_num = len(robust_acc)
x = [i for i in range(epoch_num)]
plt.plot(x, robust_acc, label = 'robust_acc')
plt.plot(x, robust_loss, label = 'robust_loss')
plt.plot(x, clean_acc, label = 'clean_acc')
plt.plot(x, clean_loss, label = 'clean_loss')
plt.legend()
plt.show()


