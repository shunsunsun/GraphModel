#!/usr/bin/env python
from scipy import sparse as sp
import argparse
import cPickle
import numpy as np
import os
import sys
import time
import tensorflow as tf
from model import Model
from utils import data

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', help='the description of dataset', type=str, default='cora')
parser.add_argument('--attr_filename', help='the attribute filename of the dataset',
                    type=str, default='graph/cora.feature')
parser.add_argument('--label_filename', help='the label filename of the dataset',
                    type=str, default='graph/cora.label')
parser.add_argument('--edge_filename', help='the edge filename of the dataset',
                    type=str, default='graph/cora.edgelist')
parser.add_argument('--labels_num', help='number of labels',
                    type=int, default=7)
parser.add_argument('--em_learning_rate',
                    help='learning rate for embedding loss', type=float, default=1e-2)
parser.add_argument('--lp_learning_rate',
                    help='learning rate for label propagation loss', type=float, default=1e-2)
parser.add_argument(
    '--ratio', help='ratio of labeled nodes for label propagation', type=float, default=0.05)
parser.add_argument('--embedding_size',
                    help='embedding dimensions', type=int, default=128)
parser.add_argument(
    '--window_size', help='window size in random walk sequences', type=int, default=5)
parser.add_argument(
    '--path_size', help='length of random walk sequences', type=int, default=80)
parser.add_argument('--graph_context_batch_size',
                    help='batch size for graph context loss', type=int, default=64)
# see the comments below.
parser.add_argument('--label_context_batch_size',
                    help='batch size for label context loss', type=int, default=256)
parser.add_argument(
    '--neg_samp', help='negative sampling rate', type=int, default=6)
# reduce this number if you just want to get some quick results. Increasing this number usually leads to better results but takes more time.
parser.add_argument(
    '--max_iter', help='max iterations of training', type=int, default=100)
parser.add_argument(
    '--mu', help='hyper-parameter for label propagation', type=float, default=10)
parser.add_argument(
    '--keep_prob', help='keep probability for dropout', type=float, default=1.0)
# In particular, $\alpha = (label_context_batch_size)/(graph_context_batch_size + label_context_batch_size)$.
# A larger value is assigned to graph_context_batch_size if graph structure is more informative than label information.


args = parser.parse_args()
print args

data = data(args)

print_every_k_iters = 1
start_time = time.time()

with tf.Session() as session:
    model = Model(args, data, session)
    iter_cnt = 0
    augument_size = int(len(model.label_x))
    while True:
        iter_cnt += 1
        curr_loss_label_propagation, curr_loss_u_1, curr_loss_u_2 = (
            0.0, 0.0, 0.0)

        average_loss_u_1 = 0
        average_loss_u_2 = 0
        for i in range(200):
            # unsupervised training using network context.
            curr_loss_u_1 = model.unsupervised_train_step(
                is_graph_context=True)
            # unsupervised training using label context.
            curr_loss_u_2 = model.unsupervised_train_step(
                is_label_context=True)
            average_loss_u_1 += curr_loss_u_1
            average_loss_u_2 += curr_loss_u_2

        average_loss_u_1 /= 200.0
        average_loss_u_2 /= 200.0

        S = model.calc_similarity()

        curr_loss_label_propagation = model.label_propagation_train_step(
            S, 100)
        entropy_loss_vec = model.calc_entropy_loss()
        entropy_loss_vec.sort()

        chosed_ranking_loss = entropy_loss_vec[int(augument_size)]
        if augument_size < model.vertices_num/2 and iter_cnt >= 10:
            model.session.run(model.lambd.assign(chosed_ranking_loss))
            augument_size += model.vertices_num/(args.max_iter * 1.0)

            if iter_cnt != 1:
                V_pre = model.session.run(model.V)

            # Update indicator vector via closed form solution
            model.indicator_vector_train_step(S)

            # Augument label context nodes
            V, F = model.session.run([model.V, model.F])

            if np.sum(V) != 0:
                nonzero_idx = np.nonzero(V)[0]
                for i in nonzero_idx:
                    if model.augumentDict.has_key(i):
                        continue
                    augument_label = np.argmax(F[i])
                    model.label2idx[augument_label].append(i)
                    for j in range(model.labels_num):
                        if j is not augument_label:
                            model.not_label2idx[j].append(i)
                    model.augumentDict[i] = 1
                    model.augument_Label_x.append(i)
                    model.augument_Label_y.append(augument_label)

        if iter_cnt % print_every_k_iters == 0:  # for printing info.
            curr_loss = average_loss_u_1 + average_loss_u_2 + curr_loss_label_propagation
            print "iter = %d, loss = %f, time = %d s" % (iter_cnt, curr_loss, int(time.time()-start_time))
            print "embedding loss: " + str(average_loss_u_1 + average_loss_u_2)
            print "label propagation loss: " + str(curr_loss_label_propagation)
            model.evaluation_label_propagation()

        if iter_cnt == args.max_iter:
            acc = model.evaluation_label_propagation()
            print "Model has converged."
            break

    model.store_useful_information()

    print "The final accuracy is: " + str(acc)
