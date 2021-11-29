from collections import defaultdict as ddict

import dgl
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import AGNNConv as AGNNConvDGL
from dgl.nn.pytorch import APPNPConv
from dgl.nn.pytorch import ChebConv as ChebConvDGL
from dgl.nn.pytorch import GATConv as GATConvDGL
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import auc, roc_auc_score
from torch import nn
from torch.nn import ELU, Dropout, Linear, ReLU, Sequential

from .MLP import MLPRegressor
from .models import BaseModel


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
        m_hot = torch.zeros(cosine1.size(), device=cosine1.device)
        if m is None:
            m_hot.scatter_(1, labels[:, None], self.m)
        ac = torch.acos(cosine1)
        ac += m_hot
        cosine = torch.cos(ac).mul_(self.s)
        return cosine


class GNNModelDGL(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        with_arcface,
        dropout=0.0,
        name="gat",
        residual=True,
        use_mlp=False,
        join_with_mlp=False,
        n_classes=None,
        s=None,
        m=None,
    ):
        super(GNNModelDGL, self).__init__()
        self.name = name
        self.use_mlp = use_mlp
        self.join_with_mlp = join_with_mlp
        self.normalize_input_columns = True

        self.with_arcface = with_arcface
        if with_arcface:
            assert s is not None, "Forgot to specify s parameter"
            assert m is not None, "Forgot to specify m parameter"
            self.arcface = ArcFace(hidden_dim, n_classes, s, m)

        if use_mlp:
            self.mlp = MLPRegressor(in_dim, hidden_dim, out_dim)
            if join_with_mlp:
                in_dim += out_dim
            else:
                in_dim = out_dim
        if name == "gat":
            self.l1 = GATConvDGL(
                in_dim,
                hidden_dim // 8,
                8,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False,
                activation=F.elu,
            )
            self.l2 = GATConvDGL(
                hidden_dim,
                out_dim,
                1,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=residual,
                activation=None,
            )
        elif name == "gcn":
            self.l1 = GraphConv(in_dim, hidden_dim, activation=F.elu)
            self.l2 = GraphConv(hidden_dim, out_dim, activation=F.elu)
            self.drop = Dropout(p=dropout)
        elif name == "cheb":
            self.l1 = ChebConvDGL(in_dim, hidden_dim, k=3)
            self.l2 = ChebConvDGL(hidden_dim, out_dim, k=3)
            self.drop = Dropout(p=dropout)
        elif name == "agnn":
            self.lin1 = Sequential(
                Dropout(p=dropout), Linear(in_dim, hidden_dim), ELU()
            )
            self.l1 = AGNNConvDGL(learn_beta=False)
            self.l2 = AGNNConvDGL(learn_beta=True)
            self.lin2 = Sequential(
                Dropout(p=dropout), Linear(hidden_dim, out_dim), ELU()
            )
        elif name == "appnp":
            self.lin1 = Sequential(
                Dropout(p=dropout),
                Linear(in_dim, hidden_dim),
                ReLU(),
                Dropout(p=dropout),
                Linear(hidden_dim, out_dim),
            )
            self.l1 = APPNPConv(k=10, alpha=0.1, edge_drop=0.0)

    def forward(self, graph, features, labels=None, emb_only=False, m=None):
        h = features
        if self.use_mlp:
            if self.join_with_mlp:
                h = torch.cat((h, self.mlp(features)), 1)
            else:
                h = self.mlp(features)
        if self.name == "gat":
            h = self.l1(graph, h).flatten(1)
            logits = self.l2(graph, h).mean(1)
        elif self.name in ["appnp"]:
            h = self.lin1(h)
            logits = self.l1(graph, h)
        elif self.name == "agnn":
            h = self.lin1(h)
            h = self.l1(graph, h)
            h = self.l2(graph, h)
            logits = self.lin2(h)
        elif self.name in ["gcn", "cheb"]:
            h = self.drop(h)
            h = self.l1(graph, h)
            logits = self.l2(graph, h)

        with graph.local_scope():
            graph.ndata["h"] = logits
            graph_embedding = dgl.mean_nodes(graph, "h")

            if emb_only:
                return graph_embedding

            if self.with_arcface:
                return self.arcface(graph_embedding, labels, m)
            else:
                return self.classify(graph_embedding)

        return logits


class GNN(BaseModel):
    def __init__(
        self,
        with_arcface,
        lr=0.01,
        hidden_dim=64,
        dropout=0.0,
        name="gat",
        residual=True,
        lang="dgl",
        gbdt_predictions=None,
        mlp=False,
        use_leaderboard=False,
        only_gbdt=False,
        s=None,
        m=None,
    ):
        super(GNN, self).__init__()

        self.with_arcface = with_arcface
        self.dropout = dropout
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.model_name = name
        self.use_residual = residual
        self.lang = lang
        self.use_mlp = mlp
        self.use_leaderboard = use_leaderboard
        self.gbdt_predictions = gbdt_predictions
        self.only_gbdt = only_gbdt
        self.s = s
        self.m = m

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    def __name__(self):
        if self.gbdt_predictions is None:
            return "GNN"
        else:
            return "ResGNN"

    def init_model(self):
        self.model = GNNModelDGL(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            with_arcface=self.with_arcface,
            dropout=self.dropout,
            name=self.model_name,
            residual=self.use_residual,
            use_mlp=self.use_mlp,
            join_with_mlp=self.use_mlp,
            n_classes=self.out_dim,
            s=self.s,
            m=self.m,
        ).to(self.device)

    def train_model(self, loader, feat_name):
        self.model.train()

        for gs, ls in loader:
            gs = gs.to(self.device)
            ls = ls.to(self.device)
            feats = gs.ndata[feat_name].float()
            logits = self.model(gs, feats, ls)
            loss = F.cross_entropy(logits, ls)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def evaluate_model(self, loader, feat_name):
        self.model.eval()

        elements = 0
        accs = 0
        losses = []
        for gs, ls in loader:
            gs = gs.to(self.device)
            ls = ls.to(self.device)
            feats = gs.ndata[feat_name].float()
            logits = self.model(gs, feats, ls, m=0).squeeze()
            losses.append(F.cross_entropy(logits, ls.long()).cpu().detach().item())
            accs += (ls == logits.max(1)[1]).sum()
            elements += ls.shape[0]

        return np.mean(losses), accs.detach().item() / elements

    def init_optimizer(self):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=3e-4
        )

    def compute_auc_roc(self, loader1, loader2, test_labels, feat_name):

        for g in loader1:
            g = g.to(self.device)
            feats = g.ndata[feat_name].float()
            emb1 = self.model(g, feats, labels=None, emb_only=True)

        for g in loader2:
            g = g.to(self.device)
            feats = g.ndata[feat_name].float()
            emb2 = self.model(g, feats, labels=None, emb_only=True)

        test_scores = (F.normalize(emb1) * F.normalize(emb2)).sum(1)
        return roc_auc_score(test_labels, test_scores.detach().cpu()), test_scores

    def find_thresholds_by_FAR(self, score_vec, label_vec, FARs=None, epsilon=1e-8):
        """
        Find thresholds given FARs
        but the real FARs using these thresholds could be different
        the exact FARs need to recomputed using calcROC
        """
        assert len(score_vec.shape) == 1
        assert score_vec.shape == label_vec.shape
        assert label_vec.dtype == np.bool
        score_neg = score_vec[~label_vec]
        score_neg[::-1].sort()
        num_neg = len(score_neg)

        assert num_neg >= 1

        if FARs is None:
            thresholds = np.unique(score_neg)
            thresholds = np.insert(thresholds, 0, thresholds[0] + epsilon)
            thresholds = np.insert(
                thresholds, thresholds.size, thresholds[-1] - epsilon
            )
        else:
            FARs = np.array(FARs)
            num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

            thresholds = []
            for num_false_alarm in num_false_alarms:
                if num_false_alarm == 0:
                    threshold = score_neg[0] + epsilon
                else:
                    threshold = score_neg[num_false_alarm - 1]
                thresholds.append(threshold)
            thresholds = np.array(thresholds)

        return thresholds

    def ROC(
        self, score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False
    ):
        """
        Compute Receiver operating characteristic (ROC) with a score and label vector.
        """
        assert score_vec.ndim == 1
        assert score_vec.shape == label_vec.shape
        assert label_vec.dtype == np.bool

        if thresholds is None:
            thresholds = self.find_thresholds_by_FAR(score_vec, label_vec, FARs=FARs)

        assert len(thresholds.shape) == 1

        # FARs would be check again
        TARs = np.zeros(thresholds.shape[0])
        FARs = np.zeros(thresholds.shape[0])
        false_accept_indices = []
        false_reject_indices = []
        for i, threshold in enumerate(thresholds):
            accept = score_vec >= threshold
            TARs[i] = np.mean(accept[label_vec])
            FARs[i] = np.mean(accept[~label_vec])
            if get_false_indices:
                false_accept_indices.append(
                    np.argwhere(accept & (~label_vec)).flatten()
                )
                false_reject_indices.append(
                    np.argwhere((~accept) & label_vec).flatten()
                )

        if get_false_indices:
            return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
        else:
            return TARs, FARs, thresholds

    def compute_auc_tar_far(
        self, scores, labels, far_upper=0.2, to_show=False, output_fn=None
    ):

        score_vec = scores.cpu().detach().numpy()
        label_vec = np.array(labels).astype(bool)
        TARs, FARs, thresholds = self.ROC(
            score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False
        )

        idxs = np.where(FARs <= far_upper)[0]
        xvals = FARs[idxs]
        yvals = TARs[idxs]

        aucscore = auc(xvals, yvals)

        if to_show:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xvals, y=yvals, mode="lines+markers"))
            fig.update_layout(
                title=f"FAR vs TAR: {aucscore:.4f}",
                title_x=0.5,
                xaxis_title="Epoch",
                yaxis_title="",
                font=dict(
                    size=40,
                ),
                #     height=600,
            )
            fig.show()
            fig.write_image(output_fn) if output_fn else None

        return aucscore, FARs, TARs

    def plot_interactive(
        self,
        metrics_list,
        legend=["Train", "Val", "Test"],
        title="",
        logx=False,
        logy=False,
        metric_name="loss",
        start_from=0,
        output_fn=None,
        to_show=True,
    ):

        fig = go.Figure()
        dash_opt = ["dash", "dot"]

        for mi, metrics in enumerate(metrics_list):
            metric_results = metrics[metric_name]
            xs = [list(range(len(metric_results)))] * len(metric_results[0])
            ys = list(zip(*metric_results))

            for i in range(len(ys)):
                fig.add_trace(
                    go.Scatter(
                        x=xs[i][start_from:],
                        y=ys[i][start_from:],
                        mode="lines+markers",
                        name=legend[i + mi * 3],
                        line={"dash": dash_opt[mi]},
                    )
                )

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title="Epoch",
            yaxis_title="",
            font=dict(
                size=40,
            ),
            height=600,
        )

        if logx:
            fig.update_layout(xaxis_type="log")
        if logy:
            fig.update_layout(yaxis_type="log")

        #     plt.savefig(output_fn, bbox_inches='tight') if output_fn else None
        fig.write_image(output_fn) if output_fn else None
        if to_show:
            fig.show()

    def save_metrics(self, metrics, fn):
        if fn is not None:
            with open(fn, "w+") as f:
                for key, value in metrics.items():
                    print(key, value, file=f)

    def fit(
        self,
        train_graphs,
        train_labels,
        test_graphs1,
        test_graphs2,
        test_labels,
        num_epochs,
        output_fn=None,
    ):

        # TODO: try this new class working

        torch.manual_seed(0)
        np.random.seed(0)

        data = list(zip(train_graphs, train_labels))
        np.random.shuffle(data)

        trainloader = GraphDataLoader(
            data[: int(0.9 * len(data))],
            batch_size=int(0.9 * len(data)),
            drop_last=False,
            shuffle=True,
        )
        testloader = GraphDataLoader(
            data[int(0.9 * len(data)) :],
            batch_size=len(data[int(0.9 * len(data)) :]),
            drop_last=False,
            shuffle=False,
        )

        testloader1 = GraphDataLoader(
            test_graphs1, batch_size=len(test_graphs1), drop_last=False, shuffle=False
        )
        testloader2 = GraphDataLoader(
            test_graphs2, batch_size=len(test_graphs2), drop_last=False, shuffle=False
        )

        feat_name = "node_attr"  # 'node_attr'
        # edge_name = None  # 'edge_labels'

        self.device = torch.device(f"cuda:{2}")
        # device = torch.device('cpu')

        self.in_dim = train_graphs[0].ndata[feat_name].shape[1]
        self.hidden_dim = 128
        self.out_dim = np.unique(train_labels).shape[0]

        far_upper = 0.2

        self.init_model()
        self.init_optimizer()

        metrics = ddict(list)

        for epoch in range(num_epochs):
            self.train_model(trainloader, feat_name)

            loss1, acc1 = self.evaluate_model(trainloader, feat_name)
            loss2, acc2 = self.evaluate_model(testloader, feat_name)

            roc_auc, test_scores = self.compute_auc_roc(
                testloader1, testloader2, test_labels, feat_name
            )
            tarfar_auc, FARs, TARs = self.compute_auc_tar_far(
                test_scores,
                test_labels,
                far_upper=far_upper,
                to_show=False,
                output_fn=None,
            )

            metrics["auc"].append((roc_auc,))
            metrics["tar_auc"].append((tarfar_auc,))
            metrics["acc"].append((acc1, acc2))
            metrics["loss"].append((loss1, loss2))

        self.save_metrics(metrics, output_fn)
