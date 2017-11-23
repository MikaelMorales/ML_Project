# -*- coding: utf-8 -*-
import numpy as np
from image_helpers import *

def compute_accurary(y, y_te):
    y_te = y_te.reshape(-1)
    y = y.reshape(-1)
    diff = y - y_te
    same = np.sum(diff == 0)
    return same / y_te.size

def build_k_indices(y, k_fold, seed=1):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
    
def cross_validation(model, y, x, k, patchSize, threshold):
    k_indices = build_k_indices(y, 4)
    res = np.zeros(k)
    
    for i in range(k):
        # Get the test fold
        x_te = x[k_indices[i]]
        y_te = y[k_indices[i]]

        # Get the train folds
        train_indices = np.concatenate([k_indices[j] for j in [v for v in range(k) if v != i]])
        x_tr = x[train_indices]
        y_tr = y[train_indices]

        model.classificationMethod() # Reset model
        model.train(y_tr, x_tr)
    
        # Run classification
        Z = model.classify(x_te)
    
        # Calculate ground-truth labels
        img_patches_gt = create_patches(y_te, patchSize)
        y_real = np.mean(img_patches_gt, axis=(1, 2)) > threshold
        
        res[i] = compute_accurary(y_real, Z)
    
    print('Cross validation accuracy: ' + str(np.mean(res)) + ', std=' + str(np.std(res)))    