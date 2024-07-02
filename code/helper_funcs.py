import numpy as np

def cal_utility(n_correct, n_wrong, t):
    n_correct = n_correct - n_wrong
    return np.log(36-1)/np.log(2)*n_correct/t
def cal_t_selection(n_flash, t0 = 3500, t1 = 31.25+125):
    return (n_flash*t1+t0)/1000/60
def evaluate_chr_accu(yhat, y):
  correct = yhat.cumsum(axis=1).argmax(axis=3) == y.argmax(axis=3)
  correct = correct.sum(axis=2) == 2
  return np.mean(correct, axis=0)

def evaluate(yhat, y):
    correct = yhat.cumsum(axis=1).argmax(axis=3) == y.argmax(axis=3)
    correct = correct.sum(axis=2) == 2
    result = {
        'accuracy': np.mean(correct, axis=0).tolist(),
        'correct': correct.tolist()
    }
    return result