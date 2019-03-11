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
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=10,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('--cost_func', type=str, default="mse_loss",
                    help='cost_func : mse_loss,l1_loss , smooth_l1_loss , ')
parser.add_argument('--trial', type=int, default=1,
                    help='trial')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, out_feature, test_in_features, test_out_features = load_my_data()

loss_function_dict = {
    'mse_loss': F.mse_loss,
    'l1_loss': F.l1_loss,
    'smooth_l1_loss': F.smooth_l1_loss
}
loss_fuction = loss_function_dict[args.cost_func]
# Model and optimizer
model = MyGCN(nfeat=features.shape[2],
              nhid=args.hidden,
              nout=out_feature.shape[2],
              dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    out_feature = out_feature.cuda()
    test_in_features = test_in_features.cuda()
    test_out_features = test_out_features.cuda()


def train(epoch):
    t = time.time()
    torch.cuda.synchronize()
    num_data = features.shape[0]
    rand_sample = np.random.randint(num_data, size=10)
    if not args.fastmode:
        model.eval()
        output = model(features[rand_sample], adj)
    else:
        model.train()
        output = model(features[rand_sample], adj)

    prev_loss = loss_fuction(output, out_feature[rand_sample])

    model.train()
    optimizer.zero_grad()

    for ind in range(num_data):
        prev_batch_ind = ind
        curr_batch_ind = ind+1

        if(ind == 0):
            batch = features[:curr_batch_ind]
            batch_out = out_feature[:curr_batch_ind]
        else:
            batch = features[prev_batch_ind:curr_batch_ind]
            batch_out = out_feature[prev_batch_ind:curr_batch_ind]

        output = model(batch, adj)
        loss_train = loss_fuction(output, batch_out)
        loss_train.backward()
        optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features[rand_sample], adj)
    else:
        output = model(features[rand_sample], adj)

    loss_val = loss_fuction(output, out_feature[rand_sample])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_prev: {:.6f}'.format(prev_loss.item()),
          'loss_curr: {:.6f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def save_tensor(trial = 1):
    path = "../result/model/" + "trial_{}".format(trial)
    torch.save(model.state_dict(), path+".pt")
    torch.save(optimizer.state_dict(), path+".opt")
    arg_dict = {
        'seed': args.seed,
        'lr': args.lr,
        'hidden': args.hidden,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'cost_func': args.cost_func,
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
    print("num hidden layer : {}".format(args.hidden))
    print("dropout          : {}".format(args.dropout))
    print("batch_size       : {}".format(args.batch_size))
    print("cost_func        : {}".format(args.cost_func))
    pass

    # Train model
t_total = time.time()
print_model_info()

for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
save_tensor(trial= args.trial)
test()
