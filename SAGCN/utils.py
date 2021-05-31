import numpy as np
import scipy.sparse as sp
import  networkx as nx
import os
import pickle
import torch
import random
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def load_data_all(dataset):

    data_path = 'data'
    suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
    objects = []
    for suffix in suffixs:
        file = os.path.join(data_path, 'ind.%s.%s'%(dataset, suffix))
        objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
    x, y, allx, ally, tx, ty, graph = objects
    x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

    # test indices
    test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
    with open(test_index_file, 'r') as f:
        lines = f.readlines()
    indices = [int(line.strip()) for line in lines]
    min_index, max_index = min(indices), max(indices)

    # preprocess test indices and combine all data
    tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
    features = np.vstack([allx, tx_extend])
    features[indices] = tx
    ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
    # ty_extend = ally[range(1015)]
    labels = np.vstack([ally, ty_extend])
    labels[indices] = ty
    c = labels[indices]

    label_matrix = np.dot(labels,labels.T)

    label_matrix = sp.coo_matrix(label_matrix,
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()

    # adj = np.matmul(adj, adj.T)
    adj = sp.coo_matrix(adj,
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)



    adj_real =sp.coo_matrix(np.multiply(adj.A,label_matrix.A),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))




    adj_cnn = normalize(sp.eye(adj.shape[0]))

    nplabel = np.where(labels)[1]
    labels_set = set(nplabel)
    label_to_indices = {label: np.where(nplabel == label)[0]
                             for label in labels_set}

    idx_train = [label_to_indices[0][random.randint(0, 10)],
                 label_to_indices[1][random.randint(0, 10)],
                 label_to_indices[2][random.randint(0, 10)],
                 label_to_indices[3][random.randint(0, 10)],
                 label_to_indices[4][random.randint(0, 10)],
                 label_to_indices[5][random.randint(0, 10)],
                 label_to_indices[6][random.randint(0, 10)]]

    # idx_train = [ 6, 54, 59,  7,  1, 47, 31]
    idx_val = range(300,800)
    idx_test = indices
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_cnn = sparse_mx_to_torch_sparse_tensor(adj_cnn)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test,adj_cnn


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index