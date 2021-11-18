import time
import numpy as np
import torch
from torch.nn import Dropout, ELU
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch import GATConv as GATConvDGL, GraphConv, ChebConv as ChebConvDGL, \
    AGNNConv as AGNNConvDGL, APPNPConv
from torch.nn import Sequential, Linear, ReLU, Identity
import dgl



class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s=4.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels, m=None):
        cosine1 = F.linear(F.normalize(inputs), F.normalize(self.weight))
#         index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(cosine1.shape, device=cosine1.device)
        if m is None:
            m_hot.scatter_(1, labels[:, None], self.m)
        ac = torch.acos(cosine1)
        ac += m_hot
        cosine = torch.cos(ac).mul_(self.s)
        return cosine

class GNNModelDGL(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, with_arcface,
                 dropout=0., name='gat', residual=True, s=None, m=None):
        super(GNNModelDGL, self).__init__()
        self.name = name

        if name == 'gat':
            self.l1 = GATConvDGL(in_dim, hidden_dim//8, 8, feat_drop=dropout, attn_drop=dropout, residual=False,
                              activation=F.elu)
            self.l2 = GATConvDGL(hidden_dim, hidden_dim, 1, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=None)
        elif name == 'gcn':
            self.l1 = GraphConv(in_dim, hidden_dim, activation=F.elu)
            self.l2 = GraphConv(hidden_dim, hidden_dim, activation=F.elu)
            self.drop = Dropout(p=dropout)
        elif name == 'cheb':
            self.l1 = ChebConvDGL(in_dim, hidden_dim, k = 3)
            self.l2 = ChebConvDGL(hidden_dim, hidden_dim, k = 3)
            self.drop = Dropout(p=dropout)
        elif name == 'agnn':
            self.lin1 = Sequential(Dropout(p=dropout), Linear(in_dim, hidden_dim), ELU())
            self.l1 = AGNNConvDGL(learn_beta=False)
            self.l2 = AGNNConvDGL(learn_beta=True)
            self.lin2 = Sequential(Dropout(p=dropout), Linear(hidden_dim, hidden_dim), ELU())
        elif name == 'appnp':
            self.lin1 = Sequential(Dropout(p=dropout), Linear(in_dim, hidden_dim),
                       ReLU(), Dropout(p=dropout), Linear(hidden_dim, hidden_dim))
            self.l1 = APPNPConv(k=10, alpha=0.1, edge_drop=0.)

        self.classify = nn.Linear(hidden_dim, out_dim)
        self.with_arcface = with_arcface
        if with_arcface:
            assert s is not None, "Forgot to specify s parameter"
            assert m is not None, "Forgot to specify m parameter"
            self.arcface = ArcFace(hidden_dim, out_dim, s, m)


    def forward(self, graph, features, labels=None, emb_only=False, m=None):
        h = features
        if self.name == 'gat':
            h = self.l1(graph, h).flatten(1)
            logits = self.l2(graph, h).mean(1)
        elif self.name in ['appnp']:
            h = self.lin1(h)
            logits = self.l1(graph, h)
        elif self.name == 'agnn':
            h = self.lin1(h)
            h = self.l1(graph, h)
            h = self.l2(graph, h)
            logits = self.lin2(h)
        elif self.name in ['gcn', 'cheb']:
            h = self.drop(h)
            h = self.l1(graph, h)
            logits = self.l2(graph, h)

        with graph.local_scope():
            graph.ndata['h'] = logits
            graph_embedding = dgl.mean_nodes(graph, 'h')

            if emb_only:
                return graph_embedding

            if self.with_arcface:
                return self.arcface(graph_embedding, labels, m)
            else:
                return self.classify(graph_embedding)

        return logits

