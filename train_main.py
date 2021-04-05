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
import argparse
import warnings
import logging
from scipy.spatial import distance
from scipy.sparse import csr_matrix, lil_matrix, load_npz, hstack, vstack, save_npz

from xclib.data import data_utils
from xclib.utils.sparse import normalize
import xclib.evaluation.xc_metrics as xc_metrics

from collections import defaultdict, Counter
from network import *
from data import *
from predict_main import *
from utils import *

torch.manual_seed(22)
torch.cuda.manual_seed_all(22)
np.random.seed(22)


def test():
    if(RUN_TYPE == "NR"):
        # introduce the tst points into the graph, assume all tst points known
        # at once. For larger graphs, doing ANNS on trn_points, labels work
        # equally well.
        tst_point_nbrs = sample_anns_nbrs(
            node_features,
            valid_tst_point_features,
            args.prediction_introduce_edges)
        val_adj_list_trn = [list(x) for x in tst_point_nbrs]

        for i, l in enumerate(val_adj_list_trn):
            for x in l:
                adjecency_lists[i + NUM_TRN_POINTS].append(x)
        new_graph = Graph(
            node_features,
            adjecency_lists,
            args.random_shuffle_nbrs)
        head_net.graph = new_graph

    t1 = time.time()
    validate(head_net, params, partition_indices, label_remapping,
             label_features, valid_tst_point_features, tst_X_Y_val, tst_exact_remove, tst_X_Y_trn, True, 100)
    print("Prediction time Per point(ms): ",
          ((time.time() - t1) / valid_tst_point_features.shape[0]) * 1000)


def train():
    if(args.mpt == 1):
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(params["num_epochs"]):

        epoch_train_start_time = time.time()
        head_net.train()
        torch.set_grad_enabled(True)

        num_batches = len(head_train_loader.dataset) // params["batch_size"]
        mean_loss = 0
        for batch_idx, batch_data in enumerate(head_train_loader):
            t1 = time.time()
            head_net.zero_grad()
            batch_size = batch_data['batch_size']

            if(args.mpt == 1):
                with torch.cuda.amp.autocast():
                    out_ans = head_net.forward(batch_data)
                    loss = head_criterion(
                        out_ans, batch_data["Y"].to(
                            out_ans.get_device()))
            elif(args.mpt == 0):
                out_ans = head_net.forward(batch_data)
                loss = head_criterion(
                    out_ans, batch_data["Y"].to(
                        out_ans.get_device()))

            if params["batch_div"]:
                loss = loss / batch_size
            mean_loss += loss.item() * batch_size

            if(args.mpt == 1):
                scaler.scale(loss).backward()  # loss.backward()
                scaler.step(head_optimizer)  # head_optimizer3.step()
                scaler.update()
            elif(args.mpt == 0):
                loss.backward()
                head_optimizer.step()
            del batch_data

        epoch_train_end_time = time.time()
        mean_loss /= len(head_train_loader.dataset)
        print(
            "Epoch: {}, loss: {}, time: {} sec".format(
                epoch,
                mean_loss,
                epoch_train_end_time -
                epoch_train_start_time))
        logging.info(
            "Epoch: {}, loss: {}, time: {} sec".format(
                epoch,
                mean_loss,
                epoch_train_end_time -
                epoch_train_start_time))

        if(epoch in params["adjust_lr_epochs"]):
            for param_group in head_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * params["dlr_factor"]

        if(val_data is not None and ((epoch == 0) or (epoch % args.validation_freq == 0) or (epoch == params["num_epochs"] - 1))):
            val_predicted_labels = lil_matrix(val_data["val_labels"].shape)

            t1 = time.time()
            with torch.set_grad_enabled(False):
                for batch_idx, batch_data in enumerate(val_data["val_loader"]):
                    val_preds, val_short = predict(head_net, batch_data)

                    if(not(val_short is None)):
                        partition_length = val_short.shape[1] // len(
                            partition_indices)
                        for i in range(1, len(partition_indices)):
                            val_short[:, i *
                                      partition_length: (i +
                                                         1) *
                                      partition_length] += partition_indices[i][0]

                        update_predicted_shortlist((batch_data["inputs"]) - _start, val_preds,
                                                   val_predicted_labels, val_short, None, 10)
                    else:
                        update_predicted(batch_data["inputs"] - _start, torch.from_numpy(val_preds),
                                         val_predicted_labels, None, 10)

            print(
                "Per point(ms): ",
                ((time.time() - t1) / val_predicted_labels.shape[0]) * 1000)
            acc, _ = run_validation(val_predicted_labels.tocsr(
            ), val_data["val_labels"], tst_exact_remove, tst_X_Y_trn, inv_prop)
            print("acc = {}".format(acc))
            logging.info("acc = {}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument(
        '--devices',
        required=True,
        help=', separated list of devices to use for training')
    parser.add_argument(
        '--save-model',
        required=True,
        type=int,
        help='whether to save trained model or not')

    parser.add_argument(
        '--num-epochs',
        required=True,
        type=int,
        help='number of epochs to train the graph(with random negatives) for')
    parser.add_argument(
        '--num-HN-epochs',
        required=True,
        type=int,
        help='number of epochs to fine tune the classifiers for')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='batch size to use')
    parser.add_argument(
        '--lr',
        required=True,
        type=float,
        help='learning rate for entire model except attention weights')
    parser.add_argument(
        '--attention-lr',
        required=True,
        type=float,
        help='learning rate for attention weights')
    parser.add_argument(
        '--adjust-lr',
        required=True,
        type=str,
        help=', separated epoch nums at which to adjust lr')
    parser.add_argument(
        '--dlr-factor',
        required=True,
        type=float,
        help='lr reduction factor')
    parser.add_argument(
        '--mpt',
        default="0",
        type=int,
        help='whether to do mixed precision training')

    parser.add_argument(
        '--restrict-edges-num',
        type=int,
        default=-1,
        help='take top neighbors when building graph')
    parser.add_argument(
        '--restrict-edges-head-threshold',
        type=int,
        default=3,
        help='take top neighbors for head labels having documents more than this')
    parser.add_argument(
        '--num-random-samples',
        required=True,
        type=int,
        help='num of batch random to sample')
    parser.add_argument(
        '--random-shuffle-nbrs',
        required=True,
        type=int,
        help='shuffle neighbors when sampling for a node')
    parser.add_argument(
        '--fanouts',
        default="3,3,3",
        type=str,
        help='fanouts for gcn')
    parser.add_argument(
        '--num-HN-shortlist',
        default=500,
        type=int,
        help='number of labels to shortlist for HN training')

    parser.add_argument(
        '--embedding-type',
        required=True,
        type=str,
        help='embedding type to use, a folder {embedding-type}CondensedData with embeddings files should be present')
    parser.add_argument(
        '--run-type',
        required=True,
        type=str,
        help='should be PR(Partial Reveal)/NR(No Reveal)')

    parser.add_argument(
        '--num-validation',
        default=25000,
        type=int,
        help='number of points to take for validation')
    parser.add_argument(
        '--validation-freq',
        default=6,
        type=int,
        help='validate after how many epochs, -1 means dont validate')
    parser.add_argument(
        '--num-shortlist',
        default=500,
        type=int,
        help='number of labels to shortlist per point for prediction')
    parser.add_argument(
        '--prediction-introduce-edges',
        default=4,
        type=int,
        help='number of edges to introduce from the test point')
    parser.add_argument(
        '--predict-ova',
        default=0,
        type=int,
        help='if to predict ova')

    parser.add_argument(
        '--A',
        default=0.55,
        type=float,
        help='param A for inv prop calculation')
    parser.add_argument(
        '--B',
        default=1.5,
        type=float,
        help='param B for inv prop calculation')

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    print("***args=", args)

    DATASET = args.dataset
    NUM_PARTITIONS = len(args.devices.strip().split(","))
    EMB_TYPE = args.embedding_type
    RUN_TYPE = args.run_type
    TST_TAKE = args.num_validation
    NUM_TRN_POINTS = -1

    #########################   Data load   #########################
    trn_point_titles = [
        line.strip() for line in open(
            "{}/trn_X.txt".format(DATASET),
            "r",
            encoding="latin").readlines()]
    tst_point_titles = [
        line.strip() for line in open(
            "{}/tst_X.txt".format(DATASET),
            "r",
            encoding="latin").readlines()]
    label_titles = [
        line.strip() for line in open(
            "{}/Y.txt".format(DATASET),
            "r",
            encoding="latin").readlines()]
    print("len(trn_point_titles), len(tst_point_titles), len(label_titles) = ", len(
        trn_point_titles), len(tst_point_titles), len(label_titles))

    trn_point_features = np.load(
        "{}/{}CondensedData/trn_point_embs.npy".format(DATASET, EMB_TYPE))
    label_features = np.load(
        "{}/{}CondensedData/label_embs.npy".format(DATASET, EMB_TYPE))
    tst_point_features = np.load(
        "{}/{}CondensedData/tst_point_embs.npy".format(DATASET, EMB_TYPE))
    print(
        "trn_point_features.shape, tst_point_features.shape, label_features.shape",
        trn_point_features.shape,
        tst_point_features.shape,
        label_features.shape)

    trn_X_Y = data_utils.read_sparse_file(
        "{}/trn_X_Y.txt".format(DATASET), force_header=True)
    tst_X_Y = data_utils.read_sparse_file(
        "{}/tst_X_Y.txt".format(DATASET), force_header=True)

    tst_valid_inds, trn_X_Y, tst_X_Y_trn, tst_X_Y_val, node_features, valid_tst_point_features, label_remapping, adjecency_lists, NUM_TRN_POINTS = prepare_data(trn_X_Y, tst_X_Y, trn_point_features, tst_point_features, label_features,
                                                                                                                                                                trn_point_titles, tst_point_titles, label_titles, args)

    hard_negs = [[] for i in range(node_features.shape[0])]

    print("trn_X_Y.shape, tst_X_Y_trn.shape, tst_X_Y_val.shape",
          trn_X_Y.shape, tst_X_Y_trn.shape, tst_X_Y_val.shape)

    temp = [
        line.strip().split() for line in open(
            "{}/filter_labels_test.txt".format(DATASET),
            "r").readlines()]
    removed = defaultdict(list)
    for x in temp:
        removed[int(float(x[0]))].append(int(float(x[1])))
    removed = dict(removed)
    del(temp)

    # remove from prediciton where label == point exactly text wise because
    # that is already removed from gt
    tst_exact_remove = {
        i: removed.get(
            tst_valid_inds[i],
            []) for i in range(
            len(tst_valid_inds))}
    print("len(tst_exact_remove)", len(tst_exact_remove))

    print("node_features.shape, len(adjecency_lists)",
          node_features.shape, len(adjecency_lists))
    graph = Graph(node_features, adjecency_lists, args.random_shuffle_nbrs)

    params = create_params_dict(
        args,
        node_features,
        trn_X_Y,
        graph,
        NUM_PARTITIONS,
        NUM_TRN_POINTS)
    print("***params=", params)

    #########################   M1/Phase1 Training(with random negatives)   ##
    head_net = GalaXCBase(params["num_labels"], params["hidden_dims"], params["devices"],
                          params["feature_dim"], params["fanouts"], params["graph"], params["embed_dims"])

    head_optimizer = torch.optim.Adam([{'params': [head_net.classifier.classifiers[0].attention_weights], 'lr': params["attention_lr"]},
                                      {"params": [param for name, param in head_net.named_parameters() if name != "classifier.classifiers.0.attention_weights"], "lr": params["lr"]}], lr=params["lr"])

    # required to split classification layer onto multiple GPUs
    partition_size = math.ceil(trn_X_Y.shape[1] / NUM_PARTITIONS)
    partition_indices = []
    for i in range(NUM_PARTITIONS):
        _start = i * partition_size
        _end = min(_start + partition_size, trn_X_Y.shape[1])
        partition_indices.append((_start, _end))

    print(partition_indices)

    val_data = create_validation_data(valid_tst_point_features, label_features, tst_X_Y_val,
                                      args, params, TST_TAKE, NUM_PARTITIONS)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="{}/GraphXML_log_{}.txt".format(DATASET, RUN_TYPE), level=logging.INFO)

    # training loop
    warnings.simplefilter('ignore')

    head_criterion = torch.nn.BCEWithLogitsLoss(reduction=params["reduction"])
    print("Model parameters: ", params)
    print("Model configuration: ", head_net)

    head_train_dataset = DatasetGraph(trn_X_Y, hard_negs)
    print('Dataset Loaded')

    hc = GraphCollator(
        head_net,
        params["num_labels"],
        params["num_random_samples"],
        num_hard_neg=0)
    print('Collator created')

    head_train_loader = torch.utils.data.DataLoader(
        head_train_dataset,
        batch_size=params["batch_size"],
        num_workers=10,
        collate_fn=hc,
        shuffle=True,
        pin_memory=False
    )

    inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, args.A, args.B)

    head_net.move_to_devices()

    if(args.mpt == 1):
        scaler = torch.cuda.amp.GradScaler()

    train()

    # should be kept as how many we want to test on
    params["num_tst"] = tst_X_Y_val.shape[0]

    if(args.save_model == 1):
        model_dir = "{}/GraphXMLModel{}".format(DATASET, RUN_TYPE)
        if not os.path.exists(model_dir):
            print("Making model dir...")
            os.makedirs(model_dir)

        torch.save(
            head_net.state_dict(),
            os.path.join(
                model_dir,
                "model_state_dict.pt"))
        with open(os.path.join(model_dir, "model_params.pkl"), "wb") as fout:
            pickle.dump(params, fout, protocol=4)

    if(params["num_HN_epochs"] <= 0):
        print("Accuracies with graph embeddings to shortlist:")
        test()
        sys.exit(
            "You have chosen not to fine tune classifiers using hard negatives by providing num_HN_epochs <= 0")

    print("==================================================================")

    #########################   M4/Phase2 Training(with hard negatives)   ####
    print("***params=", params)
    print("******  Starting HN fine tuning of calssifiers  ******")

    prediction_shortlist_trn = sample_hard_negatives(
        head_net, label_remapping, partition_indices, trn_X_Y.shape[0], params)

    head_criterion = torch.nn.BCEWithLogitsLoss(reduction=params["reduction"])
    print("Model parameters: ", params)

    head_train_dataset = DatasetGraph(trn_X_Y, prediction_shortlist_trn)
    print('Dataset Loaded')

    params["num_tst"] = 25000

    head_optimizer = torch.optim.Adam([{'params': [head_net.classifier.classifiers[0].attention_weights], 'lr': params["attention_lr"]},
                                      {"params": [param for name, param in head_net.classifier.named_parameters() if name != "classifiers.0.attention_weights"], "lr": params["lr"]}], lr=params["lr"])

    validation_freq = 1

    hc = GraphCollator(
        head_net,
        params["num_labels"],
        0,
        num_hard_neg=params["num_HN_shortlist"])
    print('Collator created')

    head_train_loader = torch.utils.data.DataLoader(
        head_train_dataset,
        batch_size=params["batch_size"],
        num_workers=6,
        collate_fn=hc,
        shuffle=True,
        pin_memory=True
    )

    inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, args.A, args.B)

    head_net.move_to_devices()

    if(args.mpt == 1):
        scaler = torch.cuda.amp.GradScaler()

    params["adjust_lr_epochs"] = np.arange(0, params["num_HN_epochs"], 4)
    params["num_epochs"] = params["num_HN_epochs"]

    train()

    print("==================================================================")
    print("Accuracies with graph embeddings to shortlist:")
    params["num_tst"] = tst_X_Y_val.shape[0]
    test()
