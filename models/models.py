import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict as ddict
from sklearn import preprocessing
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, auc
from .gnns import GNNModelDGL
import plotly.graph_objects as go
import json

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def plot(self, metrics, legend, title, output_fn=None, logx=False, logy=False, metric_name='loss'):
        import matplotlib.pyplot as plt
        metric_results = metrics[metric_name]
        xs = [range(len(metric_results))] * len(metric_results[0])
        ys = list(zip(*metric_results))

        plt.rcParams.update({'font.size': 40})
        plt.rcParams["figure.figsize"] = (20, 10)
        lss = ['-', '--', '-.', ':']
        colors = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']
        colors = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2),
                  (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
        colors = [[p / 255 for p in c] for c in colors]
        for i in range(len(ys)):
            plt.plot(xs[i], ys[i], lw=4, color=colors[i])
        plt.legend(legend, loc=1, fontsize=30)
        plt.title(title)

        plt.xscale('log') if logx else None
        plt.yscale('log') if logy else None
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.grid()
        plt.tight_layout()

        plt.savefig(output_fn, bbox_inches='tight') if output_fn else None
        plt.show()

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
            thresholds = np.insert(thresholds, thresholds.size, thresholds[-1] - epsilon)
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

    def compute_tar_far(self, score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False):
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
                false_accept_indices.append(np.argwhere(accept & (~label_vec)).flatten())
                false_reject_indices.append(np.argwhere((~accept) & label_vec).flatten())

        if get_false_indices:
            return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
        else:
            return TARs, FARs, thresholds

    def compute_auc_tar_far(self, scores, labels, far_upper=0.2, to_show=False, output_fn=None):

        score_vec = scores.cpu().detach().numpy()
        label_vec = np.array(labels).astype(bool)
        TARs, FARs, thresholds = self.compute_tar_far(score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False)

        idxs = np.where(FARs <= far_upper)[0]
        xvals = FARs[idxs]
        yvals = TARs[idxs]

        aucscore = auc(xvals, yvals)

        if to_show:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xvals, y=yvals,
                                     mode='lines+markers'))
            fig.update_layout(
                title=f'FAR vs TAR: {aucscore:.4f}',
                title_x=0.5,
                xaxis_title='Epoch',
                yaxis_title='',
                font=dict(
                    size=40,
                ),
                #     height=600,
            )
            fig.show()
            fig.write_image(output_fn) if output_fn else None

        return aucscore, FARs, TARs

    def plot_interactive(self, metrics_list, legend=['Train', 'Val', 'Test'], title='', logx=False, logy=False,
                         metric_name='loss', start_from=0, output_fn=None, to_show=True):

        fig = go.Figure()
        dash_opt = ['dash', 'dot']

        for mi, metrics in enumerate(metrics_list):
            metric_results = metrics[metric_name]
            xs = [list(range(len(metric_results)))] * len(metric_results[0])
            ys = list(zip(*metric_results))

            for i in range(len(ys)):
                fig.add_trace(go.Scatter(x=xs[i][start_from:], y=ys[i][start_from:],
                                         mode='lines+markers',
                                         name=legend[i + mi * 3], line={'dash': dash_opt[mi]}))

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title='Epoch',
            yaxis_title='',
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
                json.dump(metrics, f)



class GNN(BaseModel):
    def __init__(self, with_arcface, lr=0.01, hidden_dim=128, dropout=0.,
                 name='gat', residual=True, s=None, m=None):
        super(GNN, self).__init__()

        self.with_arcface = with_arcface
        self.dropout = dropout
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.model_name = name
        self.use_residual = residual
        self.s = s
        self.m = m

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        if self.gbdt_predictions is None:
            return 'GNN'
        else:
            return 'ResGNN'

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
        self.opt = torch.optim.Adam(self.model.parameters())

    def init_model(self):
        self.model = GNNModelDGL(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                                 with_arcface=self.with_arcface,
                                 dropout=self.dropout, name=self.model_name,
                                 residual=self.use_residual, s=self.s, m=self.m
                                 ).float().to(self.device)

    def fit(self, train_graphs, train_labels, test_graphs1, test_graphs2, test_labels,
            num_epochs, batch_size,
            output_fn=None, far_upper = 0.2, feat_name = 'node_attr'):

        torch.manual_seed(0)
        np.random.seed(0)

        data = list(zip(train_graphs, train_labels))

        np.random.shuffle(data)

        if batch_size is None:
            batch_size = int(.9 * len(data))

        trainloader = GraphDataLoader(data[:int(.9 * len(data))], batch_size=batch_size, drop_last=False,
                                      shuffle=True)
        testloader = GraphDataLoader(data[int(.9 * len(data)):], batch_size=len(data[int(.9 * len(data)):]),
                                     drop_last=False, shuffle=False)

        testloader1 = GraphDataLoader(test_graphs1, batch_size=len(test_graphs1), drop_last=False, shuffle=False)
        testloader2 = GraphDataLoader(test_graphs2, batch_size=len(test_graphs2), drop_last=False, shuffle=False)


        self.in_dim = train_graphs[0].ndata[feat_name].shape[1]
        self.out_dim = np.unique(train_labels).shape[0]

        self.init_model()
        self.init_optimizer()

        metrics = ddict(list)

        for epoch in range(num_epochs):
            self.train_model(trainloader, feat_name)

            loss1, acc1 = self.evaluate_model(trainloader, feat_name)
            loss2, acc2 = self.evaluate_model(testloader, feat_name)

            roc_auc, test_scores = self.compute_auc_roc(testloader1, testloader2, test_labels, feat_name)
            tarfar_auc, FARs, TARs = self.compute_auc_tar_far(test_scores, test_labels, far_upper=far_upper, to_show=False, output_fn=None)

            metrics['auc'].append((roc_auc,))
            metrics['tar_auc'].append((tarfar_auc,))
            metrics['acc'].append((acc1, acc2))
            metrics['loss'].append((loss1, loss2))

            if epoch % 10 == 0:
                print(epoch, metrics['auc'][-1], metrics['tar_auc'][-1], metrics['acc'][-1], metrics['loss'][-1])

        self.save_metrics(metrics, output_fn)
        return metrics