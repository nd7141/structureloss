import argparse
import json
import os
from sklearn import preprocessing
import torch
import numpy as np
import dgl
from dgl.dataloading import GraphDataLoader

from models.models import GNN
from configs import base_config

def get_train_graphs_and_labels(folder, datapart='train'):
    graphs = dgl.load_graphs(f'{folder}/{datapart}.bin')[0]
    for g in graphs:
        g.ndata['node_attr'] = g.ndata['node_attr'].float()

    labels = []
    with open(f'{folder}/{datapart}.labels') as f:
        for line in f:
            if line:
                labels.append(int(line))

    label2ix = dict()
    ix = 0
    for l in labels:
        if l not in label2ix:
            label2ix[l] = ix
            ix += 1

    labels = [label2ix[l] for l in labels]
    return graphs, labels

def get_test_graphs_and_labels(folder, datapart='test'):
    test_graphs1 = dgl.load_graphs(f'{folder}/{datapart}_graph1.bin')[0]
    test_graphs2 = dgl.load_graphs(f'{folder}/{datapart}_graph2.bin')[0]
    for g1, g2 in zip(test_graphs1, test_graphs2):
        g1.ndata['node_attr'] = g1.ndata['node_attr'].float()
        g2.ndata['node_attr'] = g2.ndata['node_attr'].float()

    test_labels = []
    with open(f'{folder}/{datapart}.labels') as f:
        for line in f:
            if line:
                test_labels.append(int(line))
    return test_graphs1, test_graphs2, test_labels

def scale_features(graphs, test_graphs1, test_graphs2):
    loader = GraphDataLoader(graphs, batch_size=len(graphs), drop_last=False, shuffle=False)
    for gs in loader:
        X = gs.ndata['node_attr']

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X)
    for g, g1, g2 in zip(graphs, test_graphs1, test_graphs2):
        g.ndata['node_attr'] = torch.tensor(min_max_scaler.transform(g.ndata['node_attr'])).float()
        g1.ndata['node_attr'] = torch.tensor(min_max_scaler.transform(g1.ndata['node_attr'])).float()
        g2.ndata['node_attr'] = torch.tensor(min_max_scaler.transform(g2.ndata['node_attr'])).float()

def load_data(folder):
    graphs, labels = get_train_graphs_and_labels(folder, datapart='train')
    test_graphs1, test_graphs2, test_labels = get_test_graphs_and_labels(folder, datapart='test')

    scale_features(graphs, test_graphs1, test_graphs2)
    return graphs, labels, test_graphs1, test_graphs2, test_labels

def create_configs(base_config, config_folder):
    config_fns = []
    for arcface in [True, False]:
        for gnn_name in ['gcn', 'gat', 'agnn']:
            exp_name = f'arcface_{arcface}-gnn_{gnn_name}.conf'
            config = base_config.copy()
            config['gnn_name'] = gnn_name
            config['with_arcface'] = arcface
            config_fn = config_folder + f'/{exp_name}'
            config_fns.append(config_fn)
            with open(config_fn, 'w+') as f:
                json.dump(config, f)

    return config_fns

def read_config(config_fn):
    with open(config_fn) as f:
        return json.load(f)

def write_plots(gnn_model, metrics, plots_exp_folder):
    gnn_model.plot_interactive([metrics], legend=['Train', 'Test'], title='TAR_FAR AUC', metric_name='tar_auc',
                               start_from=0, output_fn=plots_exp_folder+'/tar_far.png', to_show=True)
    gnn_model.plot_interactive([metrics], legend=['Train', 'Test'], title='ROC AUC', metric_name='auc', start_from=0,
                               output_fn=plots_exp_folder+'/roc.png', to_show=True)
    gnn_model.plot_interactive([metrics], legend=['Train', 'Test'], title='Train acc', metric_name='acc', start_from=0,
                               output_fn=plots_exp_folder+'/acc.png', to_show=True)
    gnn_model.plot_interactive([metrics], legend=['Train', 'Test'], title='Train loss', metric_name='loss',
                               start_from=0, output_fn=plots_exp_folder+'/loss.png', to_show=True)

def write_results(metrics, fn, exp_name):
    best_iter = np.argmax(metrics['acc'], axis=0)[-1]
    best_roc = metrics['auc'][best_iter][-1]
    best_tar_far = metrics['tar_auc'][best_iter][-1]
    best_acc = metrics['acc'][best_iter][-1]
    with open(fn, 'a+') as f:
        f.write(f"{exp_name}\t{best_roc}\t{best_tar_far}\t{best_acc}\t{best_iter}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset name", type=str)
    args = parser.parse_args()
    data_suffix = args.dataset
    assert data_suffix is not None, "You need to specify dataset via -d dataset_name"

    print(f'Running experiments for dataset {data_suffix}')

    dataset_folder = f'./data/BindingDB_{data_suffix}/'

    graphs, labels, test_graphs1, test_graphs2, test_labels = load_data(dataset_folder)

    configs_folder = dataset_folder + '/configs/'
    metrics_folder = dataset_folder + '/metrics/'
    results_folder = dataset_folder + '/results/'
    plots_folder = dataset_folder + '/plots/'

    os.makedirs(configs_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    results_fn = results_folder + f'/results.txt'
    if os.path.exists(results_fn):
        os.remove(results_fn)
        with open(results_fn, 'a+') as f:
            f.write("exp_name\tbest_roc\tbest_tar_far\tbest_acc\tbest_iter\n")

    config_fns = create_configs(base_config, configs_folder)

    # Run experiments
    for config_fn in config_fns:
        config = read_config(config_fn)
        exp_name = f"arcface_{config['with_arcface']}-gnn_{config['gnn_name']}"
        print(exp_name)

        metrics_exp_fn = metrics_folder + f'/{exp_name}.json'
        plots_exp_folder = plots_folder + f'/{exp_name}/'
        os.makedirs(plots_exp_folder, exist_ok=True)


        gnn_model = GNN(config['with_arcface'], lr=0.01, hidden_dim=128, dropout=0., name=config['gnn_name'],
                        residual=False, s=config['s'], m=config['m'])

        metrics = gnn_model.fit(graphs, labels, test_graphs1, test_graphs2, test_labels, config['num_epochs'],
                                batch_size=config['batch_size'], output_fn=metrics_exp_fn)

        write_plots(gnn_model, metrics, plots_exp_folder)
        write_results(metrics, results_fn, exp_name)
        print()

