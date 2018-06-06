from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import networkx as nx
import cPickle
import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio


class data(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.all_x = self.read_attributes(args.attr_filename)
        self.all_y = self.read_label(args.label_filename)
        self.graph = self.read_network(args.edge_filename)
        self.adj_matrix = self.gen_network_adjmatrix(args.edge_filename)

    def read_attributes(self, filename):
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        features = []
        for line in lines[1:]:
            l = line.strip("\n\r").split(" ")
            features.append(l)
        features = np.array(features, dtype=np.float32)
        features[features > 0] = 1.0  # feature binarization

        return features

    def read_attributes_mat(self, filename):
        mat = sio.loadmat(filename)
        features = mat['feature']
        features[features > 0] = 1.0

        return features

    def read_label(self, labelFile):
        # Read node label and read node label dict
        f = open(labelFile, "r")
        lines = f.readlines()
        f.close()

        labels = []
        self.labelDict = dict()

        for line in lines:
            l = line.strip("\n\r").split(" ")
            nodeID = int(l[0])
            label = int(l[1])
            labels.append(label)
            if self.labelDict.has_key(label):
                self.labelDict[label].append(nodeID)
            else:
                self.labelDict[label] = [nodeID]
        labels = np.array(labels, dtype=np.int32)
        return labels

    def read_network(self, filename):
        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        graph = dict()
        for line in lines:
            l = line.strip("\n\r").split(" ")
            node1 = int(l[0])
            node2 = int(l[1])
            if not graph.has_key(node1):
                graph[node1] = [node2]
            else:
                graph[node1].append(node2)
            if not graph.has_key(node2):
                graph[node2] = [node1]
            else:
                graph[node2].append(node1)
        return graph

    def gen_network_adjmatrix(self, filename):
        G = nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
        G = G.to_undirected()
        G_adj = nx.to_numpy_matrix(G)

        return G_adj
