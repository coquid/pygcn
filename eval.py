from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
from pygcn.utils import load_my_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from pygcn.models import *

import pickle


# Load data
adj, training_features, training_output, test_in_features, test_out_features = load_my_data(
    path="./my_data/training/")

# adj = adj.cuda()
# training_fatures = training_fatures.cuda()
# training_output = training_output.cuda()
# test_in_features = test_out_features.cuda()

cuda.current_device()
# training_features.reshape(-1, 13)

output1 = training_output.numpy().reshape(-1, 3)
output2 = training_output.numpy().reshape(-1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

for i in range(2000):
    ax1.scatter(output1[i][0], output1[i][1], output1[i][2], marker='o')
    pass

for i in range(2000):
    ax2.scatter(output2[i*3+0], output2[i*3+1], output2[i*3+2], marker='o')
    pass
# for i in range(n):
#     ax.scatter(training_features[i][0], training_features[i]
#                [1], training_features[i][2], marker='o')
#     pass

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_axis_on()
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_axis_on()
# ax.set_ybound(lower=-0.6, upper=0.6)
# ax.set_xbound(lower=-0.6, upper=0.6)
# ax.set_zbound(lower=-0.6, upper=0.6)


plt.show()
