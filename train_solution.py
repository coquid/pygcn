from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
from pygcn.utils import load_my_data, load_save_data

from pygcn.models import GCN, MyGCN_v1, MyGCN_v2

import pickle


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--trial', type=int, default=1,
                    help='trial')

args = parser.parse_args()
args.cuda = not args.no_cuda and cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features, out_feature, test_in_features, test_out_features = load_my_data(
    path="./my_data/training/",
    output_type='solution')

loss_fuction = F.mse_loss
_model = MyGCN_v1
# Model and optimizer
model = MyGCN_v1(nfeat=features.shape[2],
                 nhid=16,
                 nout=out_feature.shape[2],
                 dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model = model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    out_feature = out_feature.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()


def train(epoch):
    t = time.time()
    num_data = features.shape[0]
    num_vert = features.shape[1]
    vert_permute = np.random.permutation(num_vert)
    vert_permute = vert_permute[int(num_vert/3):]
    model.train()

    for ind in range(num_data):
        optimizer.zero_grad()
        batch = features[ind]
        batch_out = out_feature[ind]
        output = model(batch, adj)
        loss_train = loss_fuction(
            output[vert_permute], batch_out[vert_permute])
        loss_train.backward()
        optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.6f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def save_tensor(trial=1):
    path = "../result/model/" + "trial_{}".format(trial)
    torch.save(model.state_dict(), path+".pt")
    torch.save(optimizer.state_dict(), path+".opt")
    arg_dict = {
        'seed': args.seed,
        'lr': args.lr,
        'dropout': args.dropout,
    }
    f = open(path+"_arg.pkl", "wb")
    pickle.dump(arg_dict, f)
    f.close()


def test():
    model.eval()
    output = model(test_in_features, adj)
    loss_test = loss_fuction(output, test_out_features)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


def print_model_info():
    print("Model Info")
    print("seed             : {}".format(args.seed))
    print("learning rate    : {}".format(args.lr))
    print("dropout          : {}".format(args.dropout))
    pass


    # Train model
t_total = time.time()
print_model_info()

for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
save_tensor(trial=args.trial)
test()


# Export output_feature ( diff_vec )
