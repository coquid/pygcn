from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
from pygcn.utils import load_test_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from pygcn.models import *

import pickle

test_file_name = "quad"
# Load data
adj, test_in_features, test_out_features = load_test_data(
    output_type="optimal", dataset=test_file_name)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default="trial_13",
                    help='filename')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--model', type=str, default="MyGCN_v1",
                    help='MyGCN')
parser.add_argument('--hidden', type=int, default=160,
                    help='num of hidden')
args = parser.parse_args()
path = "./result/model/"
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
    adj = adj.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()

model_dict = {
    'MyGCN_v1': MyGCN_v1,
    'MyGCN_v2': MyGCN_v2,
    'MyGCN_v3': MyGCN_v3,
    'MyGCN_v6': MyGCN_v6
}

_model = model_dict[args.model]
# Model
model = _model(test_in_features.shape[2], args.hidden,
               test_out_features.shape[2], _dropout)
model.load_state_dict(torch.load(path+args.filename+".pt"))

if args.cuda:
    adj = adj.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()
    model.cuda()


def export():
    model.eval()

    test_out = np.empty((0, adj.shape[0] * test_out_features.shape[2]))
    for i in range(test_in_features.shape[0]):
        output = model(test_in_features[i], adj)
        pred = output >= 0.5
        arr = output.cpu().detach().numpy()
        arr = arr.reshape(1,
                          test_out_features.shape[1] * test_out_features.shape[2])
        test_out = np.append(test_out, arr, axis=0)

    np.savetxt(path+args.filename+"_"+test_file_name+".output_vec",
               test_out, delimiter=" ")

    with open(path+args.filename+"_"+test_file_name+".output_info", 'w') as f:
        lines = "numFrame {}\n".format(test_out.shape[0])
        f.write(lines)
        lines = "seed             : {}\n".format(_seed)
        f.write(lines)
        lines = "learning rate    : {}\n".format(_lr)
        f.write(lines)
        lines = "dropout          : {}\n".format(_dropout)
        f.write(lines)
        pass


export()


# output1 = training_output.numpy().reshape(-1, 3)
# output2 = training_output.numpy().reshape(-1)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')

# for i in range(2000):
#     ax1.scatter(output1[i][0], output1[i][1], output1[i][2], marker='o')
#     pass

# for i in range(2000):
#     ax2.scatter(output2[i*3+0], output2[i*3+1], output2[i*3+2], marker='o')
#     pass
# # for i in range(n):
# #     ax.scatter(training_features[i][0], training_features[i]
# #                [1], training_features[i][2], marker='o')
# #     pass

# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_axis_on()
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_axis_on()
# # ax.set_ybound(lower=-0.6, upper=0.6)
# # ax.set_xbound(lower=-0.6, upper=0.6)
# # ax.set_zbound(lower=-0.6, upper=0.6)


# plt.show()
