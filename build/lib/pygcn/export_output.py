from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_my_data

from pygcn.models import GCN, GCN, MyGCN_v1, MyGCN_v2, MyGCN_v3, MyGCN_v6

import pickle


test_file_name = "(1)"


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default="trial_2",
                    help='filename')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--model', type=str, default="MyGCN_v2",
                    help='MyGCN')
args = parser.parse_args()
path = "../result/model/"
with open(path+args.filename+"_arg.pkl", 'rb') as f:
    arg_dict = pickle.load(f)


_seed = arg_dict['seed']
_lr = arg_dict['lr']
_dropout = arg_dict['dropout']

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(_seed)
torch.manual_seed(_seed)
if args.cuda:
    torch.cuda.manual_seed(_seed)


# Load data
adj, test_in_feature, test_out_feature, _, _ = load_my_data()
test_in_features = test_in_feature[0:80]
test_out_features = test_out_feature[0:80]
if args.cuda:
    adj = adj.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()


model_dict = {
    'MyGCN_v1': MyGCN_v1,
    'MyGCN_v2': MyGCN_v2,
    'MyGCN_v3': MyGCN_v3,
    'MyGCN_v4': MyGCN_v4,
    'MyGCN_v5': MyGCN_v5,
    'MyGCN_v6': MyGCN_v6
}
loss_fuction = F.mse_loss
_model = model_dict[args.model]
# Model and optimizer
model = _model(test_in_features.shape[2], 20,
               test_out_features.shape[2], _dropout)
optimizer = optim.Adam(model.parameters())
model.load_state_dict(torch.load(path+args.filename+".pt"))

optimizer.load_state_dict(torch.load(path+args.filename+".opt"))
if args.cuda:
    adj = adj.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()
    model.cuda()


def export():
    model.eval()
    output = model(test_in_features, adj)
    loss_test = loss_fuction(output, test_out_features)

    arr = output.cpu().detach().numpy()
    new_arr = arr.reshape(arr.shape[0], -1)

    np.savetxt(path+args.filename+"_"+test_file_name+".output_vec",
               new_arr, delimiter=" ")

    with open(path+args.filename+"_"+test_file_name+".output_info", 'w') as f:
        lines = "numFrame {}\n".format(arr.shape[0])
        f.write(lines)
        lines = "seed             : {}\n".format(_seed)
        f.write(lines)
        lines = "learning rate    : {}\n".format(_lr)
        f.write(lines)
        lines = "dropout          : {}\n".format(_dropout)
        f.write(lines)
        lines = "Test set results :loss= {:.4f}".format(loss_test.item())
        f.write(lines)
        pass


export()
