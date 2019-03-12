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


def load_my_data(path="../data/luis/big/", dataset="hanging", num_test=10):
    """Load cloth simulation dataset"""
    print('Loading {} dataset...'.format(dataset))

    adj_list_file = path+dataset+".adj_list"
    input_npy = path+"npy/"+dataset+".input_"
    output_npy = path+"npy/"+dataset+".output_"

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

    in_features = np.empty((0, adj.shape[0], 13))
    out_features = np.empty((0, adj.shape[0], 3))
    input_ind = 1
    while True:
        try:
            in_feat = np.load(input_npy+"{}.npy".format(input_ind))
            out_feat = np.load(output_npy+"{}.npy".format(input_ind))
            in_features = np.append(in_features, in_feat, axis=0)
            out_features = np.append(out_features, out_feat, axis=0)
            input_ind += 1
        except IOError:
            break
        pass

    assert len(in_features) == len(out_features)
    p = np.random.permutation(len(in_features))
    in_features = in_features[p]
    out_features = out_features[p]
    in_features = torch.Tensor(in_features[:-num_test])
    out_features = torch.Tensor(out_features[:-num_test])
    test_in_features = torch.Tensor(in_features[-num_test:])
    test_out_features = torch.Tensor(out_features[-num_test:])

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, in_features, out_features, test_in_features, test_out_features


def load_dc_test(path="../data/luis/big/dc_test_hanging/", dataset="dc_test_hanging"):
    """Load cloth simulation dataset"""
    print('Loading {} dataset...'.format(dataset))

    adj_list_file = path+dataset+".adj_list"
    input_npy = path+"npy/"+dataset+".input"
    output_npy = path+"npy/"+dataset+".output"

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

    try:
        in_features = np.load(input_npy+".npy")
    except IOError:
        input_arr = np.loadtxt(path+dataset+".input",
                               skiprows=1, delimiter=",")
        in_features = np.reshape(
            input_arr, (input_arr.shape[0], adj.shape[0], -1))  # 총 13개 피쳐
        np.save(input_npy, in_features)
    try:
        out_features = np.load(output_npy+".npy")
    except IOError:
        output_arr = np.loadtxt(path+dataset+".output",
                                skiprows=1, delimiter=",")
        out_features = np.reshape(
            output_arr, (output_arr.shape[0], adj.shape[0], -1))  # 총 10개 피쳐
        np.save(output_npy, out_features)

    in_features = torch.Tensor(in_features)
    out_features = torch.Tensor(out_features)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, in_features, out_features


def load_save_data(path="../data/luis/big/", dataset="hanging"):
    """Load cloth simulation dataset"""
    print('Loading {} dataset...'.format(dataset))

    info_file = path+"hanging.luis_info"

    input_file = path+dataset+".input"
    output_file = path+dataset+".output"
    input_npy = path+"npy/"+dataset+".input.npy"
    output_npy = path+"npy/"+dataset+".output.npy"

    with open(info_file) as f:
        num_vert = int(f.readline().strip())
        num_feature = int(f.readline().strip())

    try:
        in_features = np.load(input_npy)
    except IOError:
        input_arr = np.loadtxt(input_file, skiprows=1, delimiter=",")
        in_features = np.reshape(
            input_arr, (input_arr.shape[0], num_vert, num_feature))  # 총 10개 피쳐
        np.save(input_npy, in_features)

    try:
        out_feature = np.load(output_npy)
    except IOError:
        output_arr = np.loadtxt(output_file, skiprows=1, delimiter=",")
        out_feature = np.reshape(
            output_arr, (output_arr.shape[0], num_vert, 3))  # 총 3개 피쳐
        np.save(output_npy, out_feature)


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
    correct = preds.sum()
    return correct / (ground_truth.shape[0]*ground_truth.shape[1]) * 100


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    load_dc_test(dataset="dc_test_hanging")
    # load_save_data(dataset="hanging_lamp")
    # load_save_data(dataset="drop_bunny_box")
    # load_save_data(dataset="hanging_bunny_box1")

# adj, features, labels, idx_train, idx_val, idx_test = load_data()
