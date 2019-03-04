import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, MyGraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class MyGCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MyGCN, self).__init__()
        self.gc1 = MyGraphConvolution(nfeat, nhid)
        self.gc2 = MyGraphConvolution(nhid, nhid)
        self.gc3 = MyGraphConvolution(nhid, nhid)
        self.gc4 = MyGraphConvolution(nhid, nhid)
        self.gc5 = MyGraphConvolution(nhid, nhid)
        self.gc6 = MyGraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        return x
