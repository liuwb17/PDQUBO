import time

import dgl
import torch
import torch.nn as nn
from .utils import fix_seed, qubo_dict_to_torch, postprocess_gnn_mis, postprocess_gnn_max_cut
from .gnn import GCN_dev, fit_model
from .instance import gen_q_dict_max_cut_sym, gen_q_dict_mis_sym

def cra(args, nx_graph, result_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fix_seed(args.seed)
    torch_type = torch.float32
    dgl_graph = dgl.from_networkx(nx_graph).to(device)

    if min([d[1] for d in nx_graph.degree()]) == 0:
        dgl_graph = dgl.add_self_loop(dgl_graph)
        print('add self loop')
    if args.task == 'mis':
        Q_mat = qubo_dict_to_torch(nx_graph, gen_q_dict_mis_sym(nx_graph, penalty=args.penalty)).to(device)
    elif args.task == 'mc':
        Q_mat = qubo_dict_to_torch(nx_graph, gen_q_dict_max_cut_sym(nx_graph)).to(device)

    # GNN Architecture
    in_feats = int(dgl_graph.number_of_nodes() ** (0.5))
    hidden_size = int(in_feats)
    num_class = 1
    dropout = 0.0
    model = GCN_dev(in_feats,
                        hidden_size,
                        num_class,
                        dropout,
                        device).to(device)
    embedding = nn.Embedding(dgl_graph.number_of_nodes(),
                             in_feats
                             ).type(torch_type).to(device)

    # Learning Parameters
    num_epoch = int(1e+5)
    lr = 1e-4
    weight_decay = 1e-2
    tol = 1e-4
    patience = 1000
    vari_param = 0
    init_reg_param = -20
    annealing_rate = 1e-3
    check_interval = 1000
    curve_rate = 2


    def loss(probs, reg_param, curve_rate=2):
        probs_ = torch.unsqueeze(probs, 1)
        # cost function
        cost = (probs_.T @ Q_mat @ probs_).squeeze()
        # annealed term
        reg_term = torch.sum(1 - (2 * probs_ - 1) ** curve_rate)
        return cost + reg_param * reg_term, cost, reg_term

    t = time.perf_counter()
    model, bit_string_CRA, cost, reg_term, runtime, incumbents, timing = fit_model(model,
                                                             dgl_graph,
                                                             embedding,
                                                             loss,
                                                             num_epoch=num_epoch,
                                                             lr=lr,
                                                             weight_decay=weight_decay,
                                                             tol=tol,
                                                             patience=patience,
                                                             device=device,
                                                             annealing=True,
                                                             init_reg_param=init_reg_param,
                                                             annealing_rate=annealing_rate,
                                                             check_interval=check_interval,
                                                             curve_rate=curve_rate,
                                                             args=args,
                                                             nx_graph=nx_graph
                                                            )
    time_spent = time.perf_counter() - t


    if args.save:
        with open(result_path, 'w') as f:
            f.write('incumbents:' + str(incumbents) + '\n')
            f.write('timing:' + str(timing) + '\n')
            f.write('total time:' + str(time_spent))