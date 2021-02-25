import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.utils.data

import numpy as np
import numba as nb
import math
import time
import os
import pickle
import random
import nmslib
import sys
from scipy.sparse import csr_matrix, lil_matrix, load_npz, hstack, vstack

from xclib.data import data_utils
from xclib.utils.sparse import normalize
import xclib.evaluation.xc_metrics as xc_metrics


def remap_label_indices(trn_point_titles, label_titles):
    label_remapping = {}
    _new_label_index = len(trn_point_titles)
    trn_title_2_index = {x: i for i, x in enumerate(trn_point_titles)}
    
    for i, x in enumerate(label_titles):
        if(x in trn_title_2_index.keys()):
            label_remapping[i] = trn_title_2_index[x]
        else:
            label_remapping[i] = _new_label_index
            _new_label_index += 1
    
    print("_new_label_index =", _new_label_index)
    return label_remapping

def make_csr_from_ll(ll, num_z):
    data = []
    indptr = [0]
    indices = []
    for x in ll:
        indices += list(x)
        data += [1.0] * len(x)
        indptr.append(len(indices))
    
    return csr_matrix((data, indices, indptr), shape=(len(ll), num_z))

@nb.njit(cache=True)
def _recall(true_labels_indices, true_labels_indptr, pred_labels_data, pred_labels_indices, pred_labels_indptr, k):
    fracs = []
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i] : true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i] : pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i] : pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:k]
        _pred_labels = _indices[top_inds]
        if(len(_true_labels) > 0):
            fracs.append(len(set(_pred_labels).intersection(set(_true_labels))) / len(_true_labels))
    return np.mean(np.array(fracs, dtype=np.float32))

def recall(true_labels, pred_labels, k):
    return _recall(true_labels.indices.astype(np.int64), true_labels.indptr, 
    pred_labels.data, pred_labels.indices.astype(np.int64), pred_labels.indptr, k)