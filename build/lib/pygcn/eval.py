from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_my_data

from pygcn.models import GCN, MyGCN

import pickle

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default="batch_size_1_lr_0.0001_dropout_0.5",
                    help='filename')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
path = "../result/model/"
with open(path+args.filename+"_arg.pkl", 'rb') as f:
    arg_dict = pickle.load(f)


_seed = arg_dict['seed']
_lr = arg_dict['lr']
_hidden = arg_dict['hidden']
_dropout = arg_dict['dropout']
_batch_size = arg_dict['batch_size']
_cost_func = arg_dict['cost_func']

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(_seed)
torch.manual_seed(_seed)
if args.cuda:
    torch.cuda.manual_seed(_seed)

# Load data
adj, _, _, test_in_features, test_out_features = load_my_data()

if args.cuda:
    adj = adj.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()

loss_function_dict = {
    'mse_loss': F.mse_loss,
    'l1_loss': F.l1_loss,
    'smooth_l1_loss': F.smooth_l1_loss
}
loss_fuction = loss_function_dict[_cost_func]
# Model and optimizer
model = MyGCN(test_in_features.shape[2], _hidden,
              test_out_features.shape[2], _dropout)
optimizer = optim.Adam(model.parameters())
model.load_state_dict(torch.load(path+args.filename+".pt"))

optimizer.load_state_dict(torch.load(path+args.filename+".opt"))
if args.cuda:
    adj = adj.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()
    model.cuda()


def test():
    output = model(test_in_features, adj)
    loss_test = loss_fuction(output, test_out_features)
    print(model.state_dict())
    print(optimizer.state_dict())
    print("seed             : {}".format(_seed))
    print("learning rate    : {}".format(_lr))
    print("num hidden layer : {}".format(_hidden))
    print("dropout          : {}".format(_dropout))
    print("batch_size       : {}".format(_batch_size))
    print("cost_func        : {}".format(_cost_func))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


test()
