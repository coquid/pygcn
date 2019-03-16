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


class MyGCN_v1(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MyGCN_v1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = (self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc7(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc8(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc10(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc11(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc12(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc13(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc14(x, adj)
        return x


class MyGCN_v1_optimal(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MyGCN_v1_optimal, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = (self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc7(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc8(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc10(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc11(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc12(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc13(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc14(x, adj)
        return F.sigmoid(x)


class MyGCN_v2(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MyGCN_v2, self).__init__()
        self.gc1 = MyGraphConvolution(nfeat, nhid)
        self.gc2 = MyGraphConvolution(nhid, nhid)
        self.gc3 = MyGraphConvolution(nhid, nhid)
        self.gc4 = MyGraphConvolution(nhid, nhid)
        self.gc5 = MyGraphConvolution(nhid, nhid)
        self.gc6 = MyGraphConvolution(nhid, nhid)
        self.gc7 = MyGraphConvolution(nhid, nhid)
        self.gc8 = MyGraphConvolution(nhid, nhid)
        self.gc9 = MyGraphConvolution(nhid, nhid)
        self.gc10 = MyGraphConvolution(nhid, nhid)
        self.gc11 = MyGraphConvolution(nhid, nhid)
        self.gc12 = MyGraphConvolution(nhid, nhid)
        self.gc13 = MyGraphConvolution(nhid, nhid)
        self.gc14 = MyGraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = (self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc6(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc7(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc8(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc10(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc11(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc12(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc13(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc14(x, adj)
        return x


class MyGCN_v3(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MyGCN_v3, self).__init__()
        self.gc1 = MyGraphConvolution(nfeat, 12)
        self.gc2 = MyGraphConvolution(12, 10)
        self.gc3 = MyGraphConvolution(10, 8)
        self.gc4 = MyGraphConvolution(8, 6)
        self.gc5 = MyGraphConvolution(6, 4)
        self.gc6 = MyGraphConvolution(4, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = (self.gc1(x, adj))
        x = F.dropout(x, p=0, training=self.training)
        x = (self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        return x


class MyGCN_v6(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MyGCN_v6, self).__init__()
        self.gc1 = MyGraphConvolution(nfeat, 12)
        self.gc2 = MyGraphConvolution(12, 11)
        self.gc3 = MyGraphConvolution(11, 10)
        self.gc4 = MyGraphConvolution(10, 9)
        self.gc5 = MyGraphConvolution(9, 8)
        self.gc6 = MyGraphConvolution(8, 7)
        self.gc7 = MyGraphConvolution(7, 6)
        self.gc8 = MyGraphConvolution(6, 5)
        self.gc9 = MyGraphConvolution(5, 4)
        self.gc10 = MyGraphConvolution(4, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = (self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc7(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc8(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc9(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc10(x, adj)
        return x


class MyGCN_batch_norm(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MyGCN_v7, self).__init__()
        self.gc1 = MyGraphConvolution(nfeat, 12)
        self.bn1 = nn.BatchNorm1d(num_features=2371)
        self.gc2 = MyGraphConvolution(12, 11)
        self.bn2 = nn.BatchNorm1d(num_features=2371)
        self.gc3 = MyGraphConvolution(11, 10)
        self.bn3 = nn.BatchNorm1d(num_features=2371)
        self.gc4 = MyGraphConvolution(10, 9)
        self.bn4 = nn.BatchNorm1d(num_features=2371)
        self.gc5 = MyGraphConvolution(9, 8)
        self.bn5 = nn.BatchNorm1d(num_features=2371)
        self.gc6 = MyGraphConvolution(8, 7)
        self.bn6 = nn.BatchNorm1d(num_features=2371)
        self.gc7 = MyGraphConvolution(7, 6)
        self.bn7 = nn.BatchNorm1d(num_features=2371)
        self.gc8 = MyGraphConvolution(6, 5)
        self.gc9 = MyGraphConvolution(5, 4)
        self.gc10 = MyGraphConvolution(4, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = (self.gc1(x, adj))
        x = self.bn1(x)
        # activation
        x = F.dropout(x, self.dropout, training=self.training)

        x = (self.gc2(x, adj))
        x = self.bn2(x)
        # activation
        x = F.dropout(x, self.dropout, training=self.training)

        x = (self.gc3(x, adj))
        x = self.bn3(x)
        # activation
        x = F.dropout(x, self.dropout, training=self.training)

        x = (self.gc4(x, adj))
        x = self.bn4(x)
        # activation
        x = F.dropout(x, self.dropout, training=self.training)

        x = (self.gc5(x, adj))
        x = self.bn5(x)
        # activation
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc6(x, adj)
        x = self.bn6(x)
        # activation
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc7(x, adj)
        x = self.bn7(x)
        # activation
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc8(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc9(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc10(x, adj)
        return x
