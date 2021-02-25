import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.utils.data

import numpy as np
import math
import time
import os
import pickle
import random
import nmslib
import sys
from scipy.sparse import csr_matrix, lil_matrix, load_npz, hstack, vstack

from torch.utils.data import IterableDataset, DataLoader


class MockGraph():
    def __init__(self, feat_data, adj_lists, random_shuffle_nbrs):
        self.feat_data = feat_data
        self.adj_lists = adj_lists
        self.random_shuffle_nbrs = random_shuffle_nbrs

    def sample_neighbors(
        self,
        nodes: np.array,
        edge_types: np.array,
        count: int = 10,
        default_node: int = -1,
        default_weight: float = 0.0,
        default_node_type: int = -1,
    ) -> (np.array, np.array, np.array, np.array):
        res = np.empty((len(nodes), count), dtype=np.int64)
        for i in range(len(nodes)):
            universe = np.array(self.adj_lists[nodes[i]], dtype=np.int64)
            
            if(self.random_shuffle_nbrs == 1):
                np.random.shuffle(universe)

            # If there are no neighbors, fill results with a dummy value.
            if len(universe) == 0:
                res[i] = np.full(count, -1, dtype=np.int64)
            else:
                repetitions = int(count / len(universe)) + 1
                res[i] = np.resize(np.tile(universe, repetitions), count)

        return (
            res,
            np.full((len(nodes), count), 0.0, dtype=np.float32),
            np.full((len(nodes), count), -1, dtype=np.int32),
            np.full((len(nodes)), 0, dtype=np.int32),
        )

    def node_features(self, nodes: np.array, features: np.array, feature_type) -> np.array:
        return torch.Tensor(self.feat_data[nodes])

class DatasetGraph(torch.utils.data.Dataset):
    def __init__(self, X_Y, hard_negs):
        self.X_Y = X_Y
        self.res_dict = [self.X_Y.indices[self.X_Y.indptr[i]: self.X_Y.indptr[i + 1]] for i in range(len(self.X_Y.indptr) - 1)]
        self.hard_negs = [list(set(hard_negs[i]) - set(self.res_dict[i])) for i in range(self.X_Y.shape[0])] 
        
        print("Shape of X_Y = ",  self.X_Y.shape, len(self.res_dict), len(hard_negs[1]), len(self.hard_negs[1]), self.res_dict[1])

    def __getitem__(self, index):
        return (index, self.res_dict[index], self.hard_negs[index])
    
    def update_hard_negs(self, hard_negs):
        self.hard_negs = [list(set(hard_negs[i]) - set(self.res_dict[i])) for i in range(self.X_Y.shape[0])]
            
    def __len__(self):
        return self.X_Y.shape[0]

class GraphCollator():
    def __init__(self, model, num_labels, num_random=0, train=1, num_hard_neg=10):
        self.model = model
        self.train = train
        self.num_hard_neg = num_hard_neg
        self.num_labels = num_labels
        self.num_random = num_random
    
    def __call__(self, batch):
        context = {}
        context["inputs"] = np.array([b[0] for b in batch], dtype=np.int64)
        self.model.query(context)
        
        if(self.train):
            all_labels_pos = [b[1] for b in batch]
            hard_neg = np.array([b[2][:self.num_hard_neg] for b in batch], dtype=np.int64)
            label_ids = np.zeros((self.num_labels, ), dtype=np.bool)
            
            label_ids[[x for subl in all_labels_pos for x in subl]] = 1
            label_ids[np.ravel(hard_neg)] = 1
            
            random_neg = np.random.choice(np.where(label_ids == 0)[0], self.num_random, replace=False)
            label_ids[random_neg] = 1
            
            label_map = {x: i for i, x in enumerate(np.where(label_ids == 1)[0])}
            
            batch_Y = np.zeros((len(batch), len(label_map)), dtype=np.float32)
            for i, labels in enumerate(all_labels_pos):
                for l in labels:
                    batch_Y[i][label_map[l]] = 1.0
                    
            context["Y"] = torch.from_numpy(batch_Y)
            context["label_ids"] = torch.tensor(label_ids)
        else:
            if(not(batch[0][1] is None)): # prediction
                if(len(batch[0]) == 2): # shortlist per point
                    context["label_ids"] = torch.LongTensor([b[1] for b in batch])
                elif(len(batch[0]) == 3): # OvA
                    context["label_ids"] = None
            else: # embeddings calc
                context["indices"] = np.array([b[2] for b in batch], dtype=np.int64)
        context['batch_size'] = len(batch)
        return context

class DatasetGraphPrediction(torch.utils.data.Dataset):
    def __init__(self, start, end, prediction_shortlist):
        self.start = start
        self.end = end
        self.prediction_shortlist = prediction_shortlist

    def __getitem__(self, index):
        if(self.prediction_shortlist is None):
            return (index + self.start, "dummy", "dummy")
        return (index + self.start, self.prediction_shortlist[index])

    def __len__(self):
        return self.end - self.start

class DatasetGraphPredictionEncode(torch.utils.data.Dataset):
    def __init__(self, nodes):
        self.nodes = nodes

    def __getitem__(self, index):
        return (self.nodes[index], None, index)

    def __len__(self):
        return len(self.nodes)
