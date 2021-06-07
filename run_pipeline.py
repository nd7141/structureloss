import time
import numpy as np
import torch
from torch.nn import Dropout, ELU
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch import GATConv
import itertools
import dgl
from collections import defaultdict as ddict
from tqdm import tqdm
import torch_geometric.data
import torch_geometric
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS, Actor
import pandas as pd
import qgrid
from collections import Counter
from sklearn.model_selection import ParameterGrid
import json
import os


np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})


class GATOptimized(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers=1,
                 heads=8,
                 activation=F.elu,
                 feat_drop=.6,
                 attn_drop=.6,
                 negative_slope=.2,
                 residual=False):
        super(GATOptimized, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dim, heads,
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                hidden_dim * heads, hidden_dim, heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            hidden_dim * heads, out_dim, 1,
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, graph, inputs):
        h = inputs
        g = graph
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


def get_web(data):
    device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu')

    graph = dgl.graph((data.data.edge_index[0], data.data.edge_index[1])).to(device)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    train_idx = np.where(data.data.train_mask[:, 0])[0]
    val_idx = np.where(data.data.val_mask[:, 0])[0]
    test_idx = np.where(data.data.test_mask[:, 0])[0]

    node_features = data.data.x.to(device)
    labels = data.data.y.to(device)

    classes = labels.unique().cpu().numpy()
    num_classes = classes.shape[0]
    num_nodes, in_dim = node_features.shape

    return graph, node_features, num_nodes, in_dim, labels, classes, num_classes, train_idx, val_idx, test_idx


def get_citation(name):
    if name.lower() == 'cora':
        data = dgl.data.CoraGraphDataset(verbose=False)
    elif name.lower() == 'citeseer':
        data = dgl.data.CiteseerGraphDataset(verbose=False)
    elif name.lower() == 'pubmed':
        data = dgl.data.PubmedGraphDataset(verbose=False)
    else:
        raise ValueError('Unknown name: {}'.format(name))

    device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu')
    graph = data[0].to(device)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    node_features = graph.ndata['feat'].to(device)
    num_nodes, in_dim = node_features.shape
    labels = graph.ndata['label'].to(device)
    classes = labels.unique().cpu().numpy()
    num_classes = classes.shape[0]
    train_idx, val_idx, test_idx = np.where(data[0].ndata['train_mask'])[0], np.where(data[0].ndata['val_mask'])[0], \
                                   np.where(data[0].ndata['test_mask'])[0]
    return graph, node_features, num_nodes, in_dim, labels, classes, num_classes, train_idx, val_idx, test_idx


def get_input(name):
    if name in ['Texas', 'Cornell', 'Wisconsin']:
        data = WebKB('./pyg', name)
        return get_web(data)
    elif name in ['Chameleon', 'Squirrel']:
        data = WikipediaNetwork('./pyg', name)
        return get_web(data)
    elif name in ['Wikics']:
        data = WikiCS('./pyg', name)
        data.data.test_mask = data.data.test_mask.unsqueeze(-1)
        return get_web(data)
    elif name in ['Actor']:
        data = Actor('./pyg')
        data.data.test_mask = data.data.test_mask.unsqueeze(-1)
        return get_web(data)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return get_citation(name)


def argmax(arr, ix):
    best_epoch = -1
    best_value = -1
    for i, el in enumerate(arr):
        if el[ix] > best_value:
            best_value = el[ix]
            best_epoch = i
    return best_value, best_epoch


def combine_inputs(features, combined_idx):
    return torch.hstack([features[list(node_idx)] for node_idx in zip(*combined_idx)])


def combine_labels(labels, label_mapping, combined_idx):
    return torch.Tensor([label_mapping[tuple(labels[i].item() for i in idx)] for idx in combined_idx])


def get_label_mapping(classes, repeat=2):
    combined_classes = list(itertools.product(classes, repeat=repeat))
    return {c: i for i, c in enumerate(combined_classes)}


def compute_accuracy(labels, logits, idx):
    pred = logits[idx]
    y = labels[idx]
    return torch.Tensor([(y == pred.max(1)[1]).sum().item() / y.shape[0]]).item()


def get_neighborhood_classes(graph, train_idx):
    us, vs = graph.edges()
    us = us.cpu().numpy()
    vs = vs.cpu().numpy()

    combined_idx = []
    for ix in range(us.shape[0]):
        if us[ix] in train_idx and vs[ix] in train_idx:
            combined_idx.append((us[ix], vs[ix]))
    return combined_idx


def plot_interactive(metrics_list, legend=['Train', 'Val', 'Test'], title='', logx=False, logy=False,
                     metric_name='loss', start_from=0):
    import plotly.graph_objects as go
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

    fig.show()


def format_lst(lst):
    return ','.join([f"{el:.3f}" for el in lst])



def get_edge_node_probs(all_logits, graph, num_classes, all_idx, pre_softmax, neighbor_mean, prob_sum):
    if pre_softmax:
        edge_label_probs = F.softmax(all_logits, dim=1)
    else:
        edge_label_probs = all_logits

    node_combined_label_probs = []
    for idx in graph.nodes():
        node_edge_probs = edge_label_probs[all_idx[:, 0] == idx]
        if neighbor_mean:
            node_edge_probs_agg = node_edge_probs.mean(0)
        else:
            node_edge_probs_agg = node_edge_probs.sum(0)
        if prob_sum:
            pair_to_ind_probs = node_edge_probs_agg.reshape((num_classes, num_classes)).sum(1)
        else:
            pair_to_ind_probs = node_edge_probs_agg.reshape((num_classes, num_classes)).mean(1)
        node_combined_label_probs.append(F.softmax(pair_to_ind_probs, dim=0))

    return torch.vstack(node_combined_label_probs)


def train_model(gnn, mlp_unary, mlp_binary, optimizer, graph, node_features, train_idx, labels, combined_idx,
                combined_labels,
                use_unary_loss, use_combined_loss, use_edge_loss, num_classes, all_idx, metrics,
                pre_softmax, neighbor_mean, prob_sum):
    gnn.train(), mlp_unary.train(), mlp_binary.train()

    h = gnn(graph, node_features).squeeze()

    unary_loss = torch.tensor(0)
    if use_unary_loss:
        unary_logits = mlp_unary(h)[train_idx]
        unary_loss = F.cross_entropy(unary_logits, labels[train_idx].long())

    combined_loss = torch.tensor(0)
    if use_combined_loss:
        combined_inputs = combine_inputs(h, combined_idx)
        combined_logits = mlp_binary(combined_inputs)
        combined_loss = F.cross_entropy(combined_logits, combined_labels.long())

    edge_loss = torch.tensor(0)
    if use_edge_loss:
        all_inputs = combine_inputs(h, all_idx)
        all_logits = mlp_binary(all_inputs)
        edge_logits = get_edge_node_probs(all_logits, graph, num_classes, all_idx, pre_softmax=pre_softmax,
                                          neighbor_mean=neighbor_mean, prob_sum=prob_sum)
        edge_loss = F.cross_entropy(edge_logits[train_idx], labels[train_idx].long())

    #     print(unary_loss.detach().item(), combined_loss.detach().item(), edge_loss.detach().item())
    loss = unary_loss + combined_loss + edge_loss

    metrics['losses'].append([unary_loss.detach().item(), combined_loss.detach().item(), edge_loss.detach().item()])

    if loss.requires_grad:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return unary_loss.detach().item(), combined_loss


def evaluate_model(gnn, mlp_unary, mlp_binary, graph, node_features, labels, idxs, metrics,
                   num_classes, all_idx, pre_softmax, neighbor_mean, prob_sum):
    gnn.eval(), mlp_unary.eval(), mlp_binary.eval()

    h = gnn(graph, node_features).squeeze()
    logits = mlp_unary(h)

    all_inputs = combine_inputs(h, all_idx)
    all_logits = mlp_binary(all_inputs)

    edge_logits = get_edge_node_probs(all_logits, graph, num_classes, all_idx, pre_softmax=pre_softmax,
                                          neighbor_mean=neighbor_mean, prob_sum=prob_sum)

    losses = []
    accs = []
    accs_edge = []
    accs_total = []
    for idx in idxs:
        y = labels[idx]
        with torch.no_grad():
            pred = logits[idx]
            pred_edge = edge_logits[idx]
            pred_total = pred + pred_edge
            losses.append(F.cross_entropy(pred, y.long()).detach().item())
            accs.append((torch.Tensor([(y == pred.max(1)[1]).sum().item() / y.shape[0]])).detach().item())
            accs_edge.append((torch.Tensor([(y == pred_edge.max(1)[1]).sum().item() / y.shape[0]])).detach().item())
            accs_total.append((torch.Tensor([(y == pred_total.max(1)[1]).sum().item() / y.shape[0]])).detach().item())

    metrics['loss'].append(losses)
    metrics['acc'].append(accs)
    metrics['acc_edge'].append(accs_edge)
    metrics['acc_total'].append(accs_total)

def run(num_epochs=10, hidden_dim=8, out_dim=64, label_repeat=2, use_unary_loss=True, use_combined_loss=True, use_edge_loss=True,
        print_epochs=10, num_runs=1, name='Texas'):
    device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu')

    graph, node_features, num_nodes, in_dim, labels, classes, num_classes, train_idx, val_idx, test_idx = get_input(
        name)
    label_mapping = get_label_mapping(classes, repeat=label_repeat)

    combined_idx = get_neighborhood_classes(graph, train_idx)

    combined_labels = combine_labels(labels, label_mapping, combined_idx).to(device)
    #     binary_labels = combine_labels(labels, label_mapping, train_idx, repeat=label_repeat).to(device)

    all_idx = get_neighborhood_classes(graph, graph.nodes())
    all_idx = torch.tensor(all_idx).to(device)

    params = {
        'pre_softmax': [False, True],
        'neighbor_mean': [False, True],
        'prob_sum': [False, True]}

    grid = ParameterGrid(params)

    grid_results = []

    for ps in grid:
        print(ps)
        pre_softmax, neighbor_mean, prob_sum = ps['pre_softmax'], ps['neighbor_mean'], ps['prob_sum']

        test_acc = ddict(list)
        for r in range(num_runs):
            metrics = ddict(list)
            print(f'Round {r}')
            gnn = GATOptimized(in_dim, hidden_dim, out_dim).to(device)
            mlp_unary = MLP(out_dim, num_classes).to(device)
            mlp_binary = MLP(2 * out_dim, len(label_mapping)).to(device)

            parameters = [gnn.parameters(), mlp_unary.parameters(), mlp_binary.parameters()]
            optimizer = torch.optim.Adam(itertools.chain(*parameters), lr=0.005, weight_decay=5e-4)

            for epoch in range(num_epochs):
                epoch_time = time.time()
                unary_loss, binary_loss = train_model(gnn, mlp_unary, mlp_binary, optimizer, graph, node_features,
                                                      train_idx, labels,
                                                      combined_idx, combined_labels,
                                                      use_unary_loss, use_combined_loss, use_edge_loss,
                                                      num_classes, all_idx, metrics, pre_softmax, neighbor_mean, prob_sum)
                evaluate_model(gnn, mlp_unary, mlp_binary, graph, node_features, labels,
                               [train_idx, val_idx, test_idx], metrics, num_classes, all_idx,
                               pre_softmax, neighbor_mean, prob_sum)

                #         pbar.set_description(f"Loss: {unary_loss + binary_loss}, Unary loss: {unary_loss}, Binary loss: {binary_loss}")
                if print_epochs and epoch % print_epochs == 0:
                    print('Epoch:', epoch, "Time:", format_lst([time.time() - epoch_time]))
                    for key in metrics:
                        print(key, metrics[key][-1])
                    print()

            for key in metrics:
                best_acc, best_epoch = argmax(metrics[key], ix=1)
                test_acc[key].append(metrics[key][best_epoch][-1])
                # print(key, best_epoch, metrics[key][best_epoch])
            print()

        print(f'{name}:')
        round_accs = []
        for key in test_acc:
            round_accs.append((round(np.mean(test_acc[key]), 4), round(np.std(test_acc[key]), 3)))
            print(f'Mean test {key} {np.mean(test_acc[key]):.3f}+-{np.std(test_acc[key]):.3f} in {num_runs} rounds')

        grid_results.append((ps, round_accs))

    if not os.path.exists('exp/pipeline/'):
        os.mkdir('exp/pipeline/')

    with open(f'exp/pipeline/{name}.json', 'w+') as f:
        json.dump(grid_results, f)



if __name__ == '__main__':
    names = ['Texas', 'Cornell', 'Wisconsin', 'Chameleon', 'Squirrel', 'Wikics', 'Actor', 'cora', 'citeseer', 'pubmed']
    use_unary_loss = True
    use_combined_loss = True
    use_edge_loss = True

    for name in names[5:]:
        print(name)
        run(num_epochs=100, hidden_dim=8, out_dim=64, label_repeat=2,
            use_unary_loss=use_unary_loss, use_combined_loss=use_combined_loss, use_edge_loss=use_edge_loss,
            print_epochs=0, num_runs=5, name=name)
