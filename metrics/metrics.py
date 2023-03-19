from scipy import stats 
import numpy as np

def calc_pearsons(preds,labels):
    r = stats.pearsonr(preds, labels)
    return r[0]

def mean_pearsons(preds,labels):
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_r = np.array([calc_pearsons(preds[:,i], labels[:,i]) for i in range(num_classes)])
    mean_r = np.mean(class_wise_r)
    return mean_r, class_wise_r