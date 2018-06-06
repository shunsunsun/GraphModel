from collections import defaultdict
from scipy import sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.metrics.pairwise import *
import argparse
import copy
import cPickle as cpkl
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.python import py_func
import warnings


class Model(object):
    def __init__(self, args, data, session):
        self.embedding_size = args.embedding_size
        self.em_learning_rate = args.em_learning_rate
        self.lp_learning_rate = args.lp_learning_rate
        self.neg_samp = args.neg_samp
        self.mu = args.mu
        self.keep_probability = args.keep_prob

        self.window_size = args.window_size
        self.path_size = args.path_size

        self.graph_context_batch_size = args.graph_context_batch_size  # rnd walk
        self.label_context_batch_size = args.label_context_batch_size  # label proximity

        self.data = data
        self.dataset = data.dataset
        self.all_x = data.all_x
        self.all_y = data.all_y
        self.graph = data.graph
        self.adj_matrix = data.adj_matrix
        self.labelDict = data.labelDict

        self.vertices_num = data.all_x.shape[0]
        self.attributes_num = data.all_x.shape[1]
        self.labels_num = args.labels_num
        self.ratio = args.ratio

        self.model_dic = dict()

        self.partial_label()

        self.unsupervised_label_contx_generator = self.unsupervised_label_context_iter()
        self.unsupervised_graph_contx_generator = self.unsupervised_graph_context_iter()
        self.session = session
        self.build_tf_graph()

    def partial_label(self):
        # fetch partial labels for training label propagation
        N = self.vertices_num
        K = self.labels_num
        ratio = self.ratio
        num = int(N * ratio)/K
        self.label_x = []
        self.label_y = []
        for k, v in self.labelDict.items():
            values = self.labelDict[k]
            np.random.shuffle(values)
            if len(values) <= num:
                for value in values:
                    self.label_x.append(value)
                    self.label_y.append(k)
            else:
                for i in range(num):
                    self.label_x.append(values[i])
                    self.label_y.append(k)

    def build_tf_graph(self):
        """Create the TensorFlow graph. """
        input_attr = tf.placeholder(
            tf.float32, shape=[None, self.attributes_num], name="input_attr")
        graph_context = tf.placeholder(
            tf.int32, shape=[None], name="graph_context")
        pos_or_neg = tf.placeholder(
            tf.float32, shape=[None], name="pos_or_neg")
        episilon = tf.constant(1e-6)
        keep_prob = tf.constant(self.keep_probability)
        input_attr_keep = tf.contrib.layers.dropout(input_attr, keep_prob)

        W_1 = tf.get_variable("W_1", shape=(self.attributes_num, 1000),
                              initializer=tf.contrib.layers.xavier_initializer())
        b_1 = tf.Variable(tf.random_uniform([1000], -1.0, 1.0))
        hidden1_layer = tf.nn.softsign(tf.matmul(input_attr_keep, W_1) + b_1)
        hidden1_layer_keep = tf.contrib.layers.dropout(
            hidden1_layer, keep_prob)

        # can add more layers here
        # W_2 layer
        W_2 = tf.get_variable("W_2", shape=(1000, 500),
                              initializer=tf.contrib.layers.xavier_initializer())
        b_2 = tf.Variable(tf.random_uniform([500], -1.0, 1.0))
        hidden2_layer = tf.nn.softsign(
            tf.matmul(hidden1_layer_keep, W_2) + b_2)
        hidden2_layer_keep = tf.contrib.layers.dropout(
            hidden2_layer, keep_prob)

        # W_3 layer
        W_3 = tf.get_variable("W_3", shape=(
            500, self.embedding_size), initializer=tf.contrib.layers.xavier_initializer())
        b_3 = tf.Variable(tf.random_uniform([self.embedding_size], -1.0, 1.0))
        embed_layer = tf.nn.softsign(tf.matmul(hidden2_layer_keep, W_3) + b_3)

        out_embed_layer = tf.get_variable("Out_embed", shape=(self.vertices_num, self.embedding_size),
                                          initializer=tf.contrib.layers.xavier_initializer())
        out_embed_vecs = tf.nn.embedding_lookup(
            out_embed_layer, graph_context)  # lookup from output

        loss_regularizer = tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3)
        loss_regularizer += tf.nn.l2_loss(b_2) + tf.nn.l2_loss(b_3)

        ele_wise_prod = tf.multiply(embed_layer, out_embed_vecs)
        loss_unsupervised = - \
            tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(
                ele_wise_prod, axis=1) * pos_or_neg) + episilon)) + 1.0 * loss_regularizer

        optimizer_unsupervised = tf.train.GradientDescentOptimizer(
            learning_rate=self.em_learning_rate).minimize(loss_unsupervised)

        self.model_dic['input'] = input_attr
        self.model_dic['truth_context'] = graph_context
        self.model_dic['pos_or_neg'] = pos_or_neg
        self.model_dic['loss_u'] = loss_unsupervised
        self.model_dic['opt_u'] = optimizer_unsupervised
        self.model_dic['embeddings'] = embed_layer

        self.F = tf.Variable(
            tf.ones([self.vertices_num, self.labels_num], dtype=tf.float32)/self.labels_num)
        self.V = tf.Variable(tf.zeros([self.vertices_num], dtype=tf.float32))
        self.lambd = tf.Variable(0.0, dtype=tf.float32)

        S = tf.placeholder(tf.float32, shape=[
                           self.vertices_num, self.vertices_num])
        self.adj_matrix = tf.convert_to_tensor(
            self.adj_matrix, dtype=tf.float32)
        S_adj = tf.multiply(S, self.adj_matrix)
        D = tf.diag(
            tf.sqrt(1.0/(tf.reduce_sum(S_adj, axis=1) + 1e-6 * tf.ones(self.vertices_num))))
        S_norm = tf.matmul(tf.matmul(D, S_adj), D)
        L = tf.eye(self.vertices_num) - S_norm

        labels = self.gen_labeled_matrix()
        labeled_F = tf.nn.embedding_lookup(self.F, self.label_x)
        labeled_Y = tf.nn.embedding_lookup(labels, self.label_x)
        F_Y = labeled_F - labeled_Y

        smooth_loss = tf.trace(
            tf.matmul(tf.matmul(tf.transpose(self.F), L), self.F))

        fitness_loss = self.mu * (tf.trace(tf.matmul(F_Y, tf.transpose(F_Y))))

        entropy_loss = tf.reduce_sum(
            tf.keras.backend.categorical_crossentropy(self.F, self.F) * self.V)

        regularizer_loss = - self.lambd * tf.reduce_sum(self.V)

        loss_labelpropagation = smooth_loss + \
            fitness_loss + entropy_loss + regularizer_loss
        optimizer_labelpropagation = tf.train.GradientDescentOptimizer(
            learning_rate=self.lp_learning_rate)

        def prox_V(grads, params, lr):
            neg_index = grads < 0
            pos_index = grads >= 0
            grads[neg_index] = (params[neg_index] - 1)/lr
            grads[pos_index] = (params[pos_index])/lr

            return grads

        def prox_F(grads, params, lr):
            params_new = params - lr * grads
            params_new_sort = - np.sort(- params_new, axis=1)

            tri = np.tri(params_new_sort.shape[1], params_new_sort.shape[1]).T
            rangeArr = np.array(range(1, params_new_sort.shape[1] + 1))
            params_new_sort_mult = (
                1 - np.dot(params_new_sort, tri))/rangeArr + params_new_sort

            params_new_sort_mult[params_new_sort_mult > 0] = 1
            params_new_sort_mult[params_new_sort_mult <= 0] = 0

            choice_index = np.argmax(params_new_sort_mult * rangeArr, axis=1)
            arr1 = np.zeros([params_new_sort.shape[1],
                             params_new_sort.shape[0]])
            arr1[choice_index, range(0, params_new_sort.shape[0])] = 1
            arr2 = 1 - arr1.cumsum(axis=0)
            arr2[choice_index, range(0, params_new_sort.shape[0])] = 1
            arr3 = np.dot(params_new_sort, arr2)
            arr4 = np.reshape((1 - arr3[range(0, params_new_sort.shape[0]), range(
                0, params_new_sort.shape[0])])/(choice_index + 1), [-1, 1])

            idx = (params_new + arr4) > 0
            temp = np.tile(arr4, (1, params_new.shape[1]))
            grads[idx] = grads[idx] - temp[idx]/lr
            grads[(params_new + arr4) <= 0] = params[(params_new + arr4) <= 0]/lr

            return grads

        def tf_prox_F(grads, params, lr):
            return py_func(prox_F, [grads, params, lr], tf.float32)

        def tf_prox_V(grads, params, lr):
            return py_func(prox_V, [grads, params, lr], tf.float32)

        grad_V = optimizer_labelpropagation.compute_gradients(
            loss_labelpropagation, self.V)
        grad_F = optimizer_labelpropagation.compute_gradients(
            loss_labelpropagation, self.F)
        apply_f_op = optimizer_labelpropagation.apply_gradients(
            [(tf_prox_F(gv[0], gv[1], self.lp_learning_rate), gv[1]) for gv in grad_F])
        apply_v_op = optimizer_labelpropagation.apply_gradients(
            [(tf_prox_V(gv[0], gv[1], self.lp_learning_rate), gv[1]) for gv in grad_V])

        self.model_dic['similarity_matrix'] = S
        self.model_dic['loss_lp'] = loss_labelpropagation
        self.model_dic['opt_lp'] = optimizer_labelpropagation
        self.model_dic['apply_v_op'] = apply_v_op
        self.model_dic['apply_f_op'] = apply_f_op

        init = tf.global_variables_initializer()
        self.session.run(init)

    def unsupervised_train_step(self, is_graph_context=False, is_label_context=False):
        """Unsupervised training step."""
        if is_graph_context:
            batch_x, batch_gy, batch_pos_or_neg = next(
                self.unsupervised_graph_contx_generator)
        elif is_label_context:
            batch_x, batch_gy, batch_pos_or_neg = next(
                self.unsupervised_label_contx_generator)
        _, loss = self.session.run([self.model_dic['opt_u'],
                                    self.model_dic['loss_u']],
                                   feed_dict={self.model_dic['input']: batch_x,
                                              self.model_dic['truth_context']: batch_gy,
                                              self.model_dic['pos_or_neg']: batch_pos_or_neg})
        return loss

    def store_useful_information(self):
        """Store useful information, such as embeddings, outlier scores, after the model finish training."""
        embeddings = self.session.run(self.model_dic['embeddings'],
                                      feed_dict={self.model_dic['input']: self.all_x})

        cpkl.dump(embeddings, open(self.data.dataset + ".embed", "wb"))

    def unsupervised_label_context_iter(self):
        self.augumentDict = dict()
        self.augument_Label_x = list(np.copy(self.label_x))
        self.augument_Label_y = list(np.copy(self.label_y))
        self.label2idx, self.not_label2idx = defaultdict(
            list), defaultdict(list)
        for i in range(len(self.label_x)):
            self.augumentDict[self.label_x[i]] = 1
            label = self.label_y[i]
            self.label2idx[label].append(self.label_x[i])
            for j in range(self.labels_num):
                if j is not label:
                    self.not_label2idx[j].append(self.label_x[i])

        while True:
            context_pairs, pos_or_neg = [], []
            cnt = 0
            while cnt < self.label_context_batch_size:
                input_idx = np.random.randint(0, len(self.augument_Label_x))
                label = self.augument_Label_y[input_idx]
                if len(self.label2idx) == 1:
                    continue
                target_idx = self.augument_Label_x[input_idx]
                context_idx = np.random.choice(self.label2idx[label])
                context_pairs.append([target_idx, context_idx])
                pos_or_neg.append(1.0)
                for _ in range(self.neg_samp):
                    context_pairs.append(
                        [target_idx, np.random.choice(self.not_label2idx[label])])
                    pos_or_neg.append(-1.0)
                cnt += 1
            context_pairs = np.array(context_pairs, dtype=np.int32)
            pos_or_neg = np.array(pos_or_neg, dtype=np.float32)
            input_idx_var = context_pairs[:, 0]
            output_idx_var = context_pairs[:, 1]
            shuffle_idx = np.random.permutation(np.arange(len(input_idx_var)))
            input_idx_var = input_idx_var[shuffle_idx]
            output_idx_var = output_idx_var[shuffle_idx]
            pos_or_neg = pos_or_neg[shuffle_idx]
            yield self.all_x[input_idx_var], output_idx_var, pos_or_neg

    def gen_rnd_walk_pairs(self):
        print "Generate random walks..."
        all_pairs = []
        permuted_idx = np.random.permutation(self.vertices_num)
        if (len(permuted_idx) > 10000):
            permuted_idx = np.random.choice(permuted_idx, 10000, replace=False)
            print "Randomly selected src nodes for random walk..."
        for start_idx in permuted_idx:
            if start_idx not in self.graph or len(self.graph[start_idx]) == 0:
                continue
            path = [start_idx]
            for _ in range(self.path_size):
                if path[-1] in self.graph:
                    path.append(np.random.choice(self.graph[path[-1]]))
            for l in range(len(path)):
                for m in range(l - self.window_size, l + self.window_size + 1):
                    if m < 0 or m >= len(path):
                        continue
                    all_pairs.append([path[l], path[m]])
        return np.random.permutation(all_pairs)

    def unsupervised_graph_context_iter(self):
        """Unsupervised graph context iterator."""
        rnd_walk_save_file = self.data.dataset + ".rnd_walks.npy"
        save_walks = np.array([], dtype=np.int32).reshape(0, 2)
        if os.path.exists(rnd_walk_save_file):
            save_walks = np.load(rnd_walk_save_file)

        all_pairs = save_walks
        new_walks = np.array([], dtype=np.int32).reshape(
            0, 2)  # buffer storage
        max_num_pairs = max(10000000, self.vertices_num * 100)
        while True:
            if len(all_pairs) == 0:
                # enough rnd walks, reuse them.
                if len(save_walks) >= max_num_pairs:
                    all_pairs = save_walks
                else:
                    all_pairs = self.gen_rnd_walk_pairs()
                    print "newly generated rnd walks " + str(all_pairs.shape)
                    # save the new walks to buffer
                    new_walks = np.concatenate((new_walks, all_pairs), axis=0)
                    if len(new_walks) >= 10000:  # buffer full.
                        save_walks = np.concatenate(
                            (save_walks, new_walks), axis=0)
                        np.save(rnd_walk_save_file, save_walks)
                        print "Successfully save the walks..."
                        new_walks = np.array([], dtype=np.int32).reshape(0, 2)
            i = 0
            j = i + self.graph_context_batch_size
            while j < len(all_pairs):
                pos_or_neg = np.array([1.0] * self.graph_context_batch_size + [-1.0]
                                      * self.graph_context_batch_size * self.neg_samp, dtype=np.float32)
                context_pairs = np.zeros(
                    (self.graph_context_batch_size + self.graph_context_batch_size * self.neg_samp, 2), dtype=np.int32)
                context_pairs[:self.graph_context_batch_size,
                              :] = all_pairs[i:j, :]
                context_pairs[self.graph_context_batch_size:, 0] = np.repeat(
                    all_pairs[i:j, 0], self.neg_samp)
                context_pairs[self.graph_context_batch_size:, 1] = np.random.randint(
                    0, self.vertices_num, size=self.graph_context_batch_size * self.neg_samp)
                input_idx_var = context_pairs[:, 0]
                output_idx_var = context_pairs[:, 1]
                shuffle_idx = np.random.permutation(
                    np.arange(len(input_idx_var)))
                input_idx_var = input_idx_var[shuffle_idx]
                output_idx_var = output_idx_var[shuffle_idx]
                pos_or_neg = pos_or_neg[shuffle_idx]
                yield self.all_x[input_idx_var], output_idx_var, pos_or_neg
                i = j
                j = i + self.graph_context_batch_size
            all_pairs = []

    def get_embeddings(self):
        embeddings = self.session.run(self.model_dic['embeddings'],
                                      feed_dict={self.model_dic['input']: self.all_x})

        return embeddings

    def calc_similarity(self, model="cosine"):
        X = self.get_embeddings()
        if model == "cosine":
            X = cosine_similarity(X)
            X = np.exp(-1/0.03 * (1 - X))
        elif model == "distance":
            X = pairwise_distances(X)
            X = 1/(1+X)

        return X

    def gen_labeled_matrix(self):
        N = self.vertices_num
        K = self.labels_num
        Y = np.zeros((N, K))
        for i in range(len(self.label_x)):
            idx = self.label_x[i]
            label = self.label_y[i]
            Y[idx, label] = 1.0
        return tf.convert_to_tensor(Y, dtype=tf.float32)

    def label_propagation_train_step(self, S, num_iterations=100):
        average_loss = 0
        for i in range(num_iterations):
            _, loss = self.session.run([self.model_dic['apply_f_op'], self.model_dic['loss_lp']], feed_dict={
                self.model_dic['similarity_matrix']: S})
            average_loss += loss
        average_loss /= num_iterations * 1.0

        return average_loss

    def indicator_vector_train_step(self, S):
        _ = self.session.run(self.model_dic['apply_v_op'], feed_dict={
                             self.model_dic['similarity_matrix']: S})

    def calc_entropy_loss(self):
        entropy = tf.keras.backend.categorical_crossentropy(self.F, self.F)
        entropy_loss_vec = self.session.run(entropy)

        return entropy_loss_vec

    def evaluation_label_propagation(self):
        F = self.session.run(self.F)
        testID = list(set(range(self.vertices_num)) - set(self.label_x))

        predict = np.argmax(F, axis=1)

        y_pred = predict[self.label_x]
        y_gt = self.all_y[self.label_x]

        accuracy = accuracy_score(y_gt, y_pred)
        macro_f1 = f1_score(y_gt, y_pred, average="macro")
        micro_f1 = f1_score(y_gt, y_pred, average="micro")

        print "Train Set, accuracy: %f, macro_f1: %f, micro_f1: %f" % (accuracy, macro_f1, micro_f1)

        y_pred = predict[testID]
        y_gt = self.all_y[testID]

        accuracy = accuracy_score(y_gt, y_pred)
        macro_f1 = f1_score(y_gt, y_pred, average="macro")
        micro_f1 = f1_score(y_gt, y_pred, average="micro")

        print "Test Set, accuracy: %f, macro_f1: %f, micro_f1: %f" % (accuracy, macro_f1, micro_f1)

        return accuracy
