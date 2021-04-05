from __future__ import print_function
from __future__ import division

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

from xclib.data import data_utils
from xclib.utils.sparse import normalize
import xclib.evaluation.xc_metrics as xc_metrics

from data import *
from utils import *
from network import HNSW


def predict(net, pred_batch):
    """
    head shorty None means predict OvA on head
    """
    net.eval()
    torch.set_grad_enabled(False)

    out_ans = net.forward(pred_batch, False)
    out_ans = out_ans.detach().cpu().numpy()
    if(pred_batch["label_ids"] is None):
        return out_ans, None
    return out_ans, pred_batch["label_ids"].detach().cpu().numpy()


def update_predicted(row_indices, predicted_batch_labels,
                     predicted_labels, remapping, top_k):
    batch_size = row_indices.shape[0]
    top_values, top_indices = predicted_batch_labels.topk(
        k=top_k, dim=1, sorted=False)
    ind = np.zeros((top_k * batch_size, 2), dtype=np.int64)
    ind[:, 0] = np.repeat(row_indices, [top_k] * batch_size)
    if(remapping is not None):
        ind[:, 1] = [remapping[x]
                     for x in top_indices.cpu().numpy().flatten('C')]
    else:
        ind[:, 1] = [x for x in top_indices.cpu().numpy().flatten('C')]
    vals = top_values.cpu().detach().numpy().flatten('C')
    predicted_labels[ind[:, 0], ind[:, 1]] = vals


def update_predicted_shortlist(
        row_indices, predicted_batch_labels, predicted_labels, shortlist, remapping, top_k=10):
    if(len(predicted_batch_labels.shape) == 1):
        predicted_batch_labels = predicted_batch_labels[None, :]
    m = predicted_batch_labels.shape[0]

    top_indices = np.argsort(predicted_batch_labels, axis=1)[
        :, ::-1][:, :top_k]
    top_values = predicted_batch_labels[np.arange(m)[:, None], top_indices]

    batch_size, shortlist_size = shortlist.shape
    ind = np.zeros((top_k * batch_size, 2), dtype=np.int)
    ind[:, 0] = np.repeat(row_indices, [top_k] * batch_size)

    if(remapping is not None):
        ind[:, 1] = [remapping[x]
                     for x in np.ravel(shortlist[np.arange(m)[:, None], top_indices])]
    else:
        ind[:, 1] = [x for x in np.ravel(
            shortlist[np.arange(m)[:, None], top_indices])]

    predicted_labels[ind[:, 0], ind[:, 1]] = np.ravel(top_values)


def run_validation(val_predicted_labels, tst_X_Y_val,
                   tst_exact_remove, tst_X_Y_trn, inv_prop):
    data = []
    indptr = [0]
    indices = []
    for i in range(val_predicted_labels.shape[0]):
        _indices1 = val_predicted_labels.indices[val_predicted_labels.indptr[i]: val_predicted_labels.indptr[i + 1]]
        _vals1 = val_predicted_labels.data[val_predicted_labels.indptr[i]: val_predicted_labels.indptr[i + 1]]

        _indices, _vals = [], []
        for _ind, _val in zip(_indices1, _vals1):
            if (_ind not in tst_exact_remove[i]) and (
                    _ind not in tst_X_Y_trn.indices[tst_X_Y_trn.indptr[i]: tst_X_Y_trn.indptr[i + 1]]):
                _indices.append(_ind)
                _vals.append(_val)

        indices += list(_indices)
        data += list(_vals)
        indptr.append(len(indices))

    _pred = csr_matrix(
        (data, indices, indptr), shape=(
            val_predicted_labels.shape))

    print(tst_X_Y_val.shape, _pred.shape)
    acc = xc_metrics.Metrics(tst_X_Y_val, inv_psp=inv_prop)
    acc = acc.eval(_pred, 5)
    _recall = recall(tst_X_Y_val, _pred, 100)
    return (acc, _recall), _pred


def encode_nodes(net, context):
    net.eval()
    torch.set_grad_enabled(False)

    embed3 = net.third_layer_enc(context["encoder"])
    embed2 = net.second_layer_enc(context["encoder"]["node_feats"])
    embed1 = net.first_layer_enc(
        context["encoder"]["node_feats"]["node_feats"])

#     embed = torch.stack((net.transform1(embed1.t()), net.transform2(embed2.t()), net.transform3(embed3.t())), dim=1)
    embed = torch.stack((embed1.t(), embed2.t(), embed3.t()), dim=1)
    embed = torch.mean(embed, dim=1)

    return embed


def validate(head_net, params, partition_indices, label_remapping,
             label_embs, tst_point_embs, tst_X_Y_val, tst_exact_remove, tst_X_Y_trn, use_graph_embs, topK):
    _start = params["num_trn"]
    _end = _start + params["num_tst"]

    if(use_graph_embs):
        label_nodes = [label_remapping[i] for i in range(len(label_remapping))]

        val_dataset = DatasetGraphPredictionEncode(label_nodes)
        hce = GraphCollator(head_net, params["num_labels"], None, train=0)
        encode_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=500,
            num_workers=10,
            collate_fn=hce,
            shuffle=False,
            pin_memory=True)

        label_embs_graph = np.zeros(
            (len(label_nodes), params["hidden_dims"]), dtype=np.float32)
        cnt = 0
        for batch in encode_loader:
            # print (len(label_nodes), cnt*512)
            cnt += 1
            encoded = encode_nodes(head_net, batch)
            encoded = encoded.detach().cpu().numpy()
            label_embs_graph[batch["indices"]] = encoded

        val_dataset = DatasetGraphPredictionEncode(
            [i for i in range(_start, _end)])
        hce = GraphCollator(head_net, params["num_labels"], None, train=0)
        encode_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=500,
            num_workers=10,
            collate_fn=hce,
            shuffle=False,
            pin_memory=True)

        tst_point_embs_graph = np.zeros(
            (params["num_tst"], params["hidden_dims"]), dtype=np.float32)
        for batch in encode_loader:
            encoded = encode_nodes(head_net, batch)
            encoded = encoded.detach().cpu().numpy()
            tst_point_embs_graph[batch["indices"]] = encoded

        label_features = label_embs_graph
        tst_point_features = tst_point_embs_graph
    else:
        label_features = label_embs
        tst_point_features = tst_point_embs[:params["num_tst"]]

    prediction_shortlists = []
    BATCH_SIZE = 2000000

    t1 = time.time()
    for i in range(len(partition_indices)):
        print("building ANNS for partition = ", i)
        label_NGS = HNSW(
            M=100,
            efC=300,
            efS=params["num_shortlist"],
            num_threads=24)
        label_NGS.fit(
            label_features[partition_indices[i][0]: partition_indices[i][1]])
        print("Done in ", time.time() - t1)
        t1 = time.time()

        tst_label_nbrs = np.zeros(
            (tst_point_features.shape[0],
             params["num_shortlist"]),
            dtype=np.int64)
        for i in range(0, tst_point_features.shape[0], BATCH_SIZE):
            print(i)
            _tst_label_nbrs, _ = label_NGS.predict(
                tst_point_features[i: i + BATCH_SIZE], params["num_shortlist"])
            tst_label_nbrs[i: i + BATCH_SIZE] = _tst_label_nbrs

        prediction_shortlists.append(tst_label_nbrs)
        print("Done in ", time.time() - t1)
        t1 = time.time()

    if(len(partition_indices) == 1):
        prediction_shortlist = prediction_shortlists[0]
    else:
        prediction_shortlist = np.hstack(prediction_shortlists)
    print(prediction_shortlist.shape)

    del(prediction_shortlists)

    val_dataset = DatasetGraphPrediction(_start, _end, prediction_shortlist)
    hcp = GraphCollator(head_net, params["num_labels"], None, train=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=10,
        collate_fn=hcp,
        shuffle=False,
        pin_memory=True)

    val_data = dict(val_labels=tst_X_Y_val[:params["num_tst"], :],
                    val_loader=val_loader)

    val_predicted_labels = lil_matrix(val_data["val_labels"].shape)

    with torch.set_grad_enabled(False):
        for batch_idx, batch_data in enumerate(val_data["val_loader"]):
            val_preds, val_short = predict(head_net, batch_data)

            partition_length = val_short.shape[1] // len(partition_indices)
            for i in range(1, len(partition_indices)):
                val_short[:, i *
                          partition_length: (i +
                                             1) *
                          partition_length] += partition_indices[i][0]

            update_predicted_shortlist((batch_data["inputs"]) - _start, val_preds,
                                       val_predicted_labels, val_short, None, topK)

    acc, _ = run_validation(val_predicted_labels.tocsr(
    ), val_data["val_labels"], tst_exact_remove, tst_X_Y_trn, params["inv_prop"])
    print("acc = {}".format(acc))
