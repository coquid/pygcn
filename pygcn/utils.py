import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx


def load_my_data(path="../my_data/training/", dataset="quad", num_test=20, output_type="solution"):
    """Load cloth simulation dataset"""
    print('Loading {} dataset...'.format(dataset))

    adj_list_file = path+output_type+"/quad.adj_list"
    input_npy = path+output_type+"/npy/"+dataset
    output_npy = path+output_type+"/npy/"+dataset

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

    out_feature_info = {
        'solution': 3,
        'unary': 1,
        'optimal': 1
    }
    num_feature = 14
    num_out_feature = out_feature_info[output_type]
    in_features = np.empty((0, adj.shape[0], num_feature))
    out_features = np.empty((0, adj.shape[0], num_out_feature))
    input_ind = 1
    while True:
        try:
            in_feat = np.load(input_npy+" ({}).npy".format(input_ind))
            input_ind += 1
            out_feat = np.load(output_npy+" ({}).npy".format(input_ind))
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


def load_save_data(path="../my_data/training/", dataset="filename", data_type="solution"):
    """Load cloth simulation dataset"""
    print('Loading {} dataset...'.format(dataset))

    info_file = path+data_type+"/quad.luis_info"

    input_file = path+data_type+"/txt/"+dataset+".input"
    output_file = path+data_type+"/txt/"+dataset+".output"
    input_npy = path+data_type+"/npy/"+dataset+".input.npy"
    output_npy = path+data_type+"/npy/"+dataset+".output.npy"

    with open(info_file) as f:
        num_vert = int(f.readline().strip())
        num_feature = int(f.readline().strip())
    num_feature = 14
    out_feature_info = {
        'solution': 3,
        'unary': 1,
        'optimal': 1
    }
    num_out_feature = out_feature_info[data_type]
    try:
        in_features = np.load(input_npy)
    except IOError:
        input_arr = np.loadtxt(input_file, skiprows=1, delimiter=",")
        in_features = np.reshape(
            input_arr, (input_arr.shape[0], num_vert, num_feature))
        np.save(input_npy, in_features)

    try:
        out_feature = np.load(output_npy)
    except IOError:
        if(data_type == 'unary'):
            output_arr = np.loadtxt(output_file, skiprows=1, delimiter=",")
            out_feature = np.reshape(
                output_arr, (output_arr.shape[0], num_out_feature))
            np.save(output_npy, out_feature)
        else:
            output_arr = np.loadtxt(output_file, skiprows=1, delimiter=",")
            out_feature = np.reshape(
                output_arr, (output_arr.shape[0], num_vert, num_out_feature))  # 총 3개 피쳐
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


def accuracy_optimal(output, labels):
    pred = output >= 0.5
    truth = labels >= 0.5
    acc = pred.eq(truth).double().sum() / len(labels)
    return acc


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    # load_dc_test(dataset="dc_test_hanging")
    # load_save_data(dataset="drop_bunny_box")
    # load_save_data(dataset="hanging_bunny_box")
    # load_save_data(dataset="hanging_lamp_v2")
    # load_save_data(dataset="hanging_lamp_v3")
    # load_save_data(dataset="hanging_lamp_v4")
    # load_save_data(dataset="hanging_lamp_v5")
    # load_save_data(dataset="hanging_lamp_v6")

    # save data
    load_save_data(path="../my_data/training/",
                   dataset="drop_bunny_box",    data_type='unary')
    load_save_data(path="../my_data/training/",
                   dataset="hanging_bunny_box", data_type='unary')
    load_save_data(path="../my_data/training/",
                   dataset="hanging_lamp",      data_type='unary')
    load_save_data(path="../my_data/training/",
                   dataset="hanging_lamp_v2",   data_type='unary')
    load_save_data(path="../my_data/training/",
                   dataset="hanging_lamp_v3",   data_type='unary')
    load_save_data(path="../my_data/training/",
                   dataset="hanging_lamp_v4",   data_type='unary')
    load_save_data(path="../my_data/training/",
                   dataset="hanging_lamp_v5",   data_type='unary')
    load_save_data(path="../my_data/training/",
                   dataset="hanging_lamp_v6",   data_type='unary')
    # adj, features, labels, idx_train, idx_val, idx_test = load_data()
