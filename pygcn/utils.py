import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_my_data(path="../data/luis/", dataset="hanging"):
    """Load cloth simulation dataset"""
    print('Loading {} dataset...'.format(dataset))

    adj_list_file = path+dataset+".adj_list"
    input_file = path+dataset+".input"
    output_file = path+dataset+".output"
    # info_file = path+dataset+".luis_info"

    # adj Matrix
    graph = {}
    with open(adj_list_file) as f:
        for line in f:
            if(line.isspace() == False):
                key, temp_val = line.strip().split(":")
                val = temp_val.strip().split(" ")
                graph[int(key)] = list(map(int, val))

    csr_adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.coo_matrix(csr_adj)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    input_arr = np.loadtxt(input_file, skiprows=1, delimiter=",")
    # input_arr = np.reshape(input_arr,(input_arr.shape[0],adj.shape[0],10))  # 총 10개 피쳐
    # in_features = np.zeros((input_arr.shape[0]),dtype=np.matrix)
    # for i in range(input_arr.shape[0]):
    #     in_features[i] = np.asmatrix(input_arr[i])
    in_features = np.reshape(
        input_arr, (input_arr.shape[0], adj.shape[0], 10))  # 총 10개 피쳐

    output_arr = np.loadtxt(output_file, skiprows=1, delimiter=",")
    # output_arr = np.reshape(output_arr,(output_arr.shape[0],adj.shape[0],3))    # 총 3개 피쳐
    # out_feature = np.zeros((output_arr.shape[0]),dtype=np.matrix)
    # for i in range(output_arr.shape[0]):
    #     out_feature[i] = np.asmatrix(output_arr[i])
    out_feature = np.reshape(
        output_arr, (output_arr.shape[0], adj.shape[0], 3))    # 총 3개 피쳐

    in_features = torch.Tensor((in_features))
    out_feature = torch.Tensor((out_feature))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = range(25)
    idx_val = range(25)
    idx_test = range(25, 30)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, in_features, out_feature, idx_train, idx_val, idx_test


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


def my_accuracy(output, ground_truth):
    preds = torch.abs(output-ground_truth)
    preds = torch.div(preds, ground_truth)
    preds = torch.mul(preds, 100)
    correct = preds.sum()
    return correct / len(ground_truth)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = load_my_data()
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
