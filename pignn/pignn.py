import dgl
import torch
import random
import os
import numpy as np
from time import time
from .utils import get_gnn_mc, get_gnn_mis, run_gnn_training_mc, run_gnn_training_mis, postprocess_gnn_mis, gen_q_matrix_mc, gen_q_matrix_mis


def pignn(args, nx_graph, result_path):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    seed_value = args.seed
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    # NN learning hypers #
    number_epochs = int(1e5)
    if args.task == 'mis':
        learning_rate = 1e-4
    elif args.task == 'mc':
        learning_rate = 1e-2
    PROB_THRESHOLD = 0.5

    # Early stopping to allow NN to train to near-completion
    tol = 1e-4  # loss must change by more than tol, or trigger
    patience = 100  # number early stopping triggers before breaking loop
    n = nx_graph.number_of_nodes()
    degrees = nx_graph.degree()
    min_degree = min([d[1] for d in degrees])

    # Establish dim_embedding and hidden_dim values
    if args.task == 'mc':
        if n > 1e5:
            dim_embedding = round(np.sqrt(n))
        else:
            dim_embedding = round(np.cbrt(n))
    else:
        dim_embedding = round(np.sqrt(n))
    hidden_dim = round(dim_embedding / 2)  # e.g. 5

    # get DGL graph from networkx graph, load onto device
    graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
    graph_dgl = graph_dgl.to(args.TORCH_DEVICE)
    if min_degree == 0:
        graph_dgl = dgl.add_self_loop(graph_dgl)
    # Construct Q matrix for graph
    # if args.task == 'mis':
    #     q_torch = qubo_dict_to_torch(nx_graph, gen_q_dict_mis(nx_graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)
    # elif args.task == 'mc':
    #     q_torch = gen_q_matrix_mc(nx_graph)
    # Establish pytorch GNN + optimizer
    opt_params = {'lr': learning_rate}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.1,
        'number_classes': 1,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience
    }




    if args.task == 'mis':
        gnn_start = time()
        q_torch = gen_q_matrix_mis(args, nx_graph)

        net, embed, optimizer = get_gnn_mis(n, gnn_hypers, opt_params, args.TORCH_DEVICE, args.TORCH_DTYPE)

        # For tracking hyperparameters in results object
        gnn_hypers.update(opt_params)

        _, epoch, final_bitstring, best_bitstring, incumbents, timing = run_gnn_training_mis(args, nx_graph,
            q_torch, graph_dgl, net, embed, optimizer, gnn_hypers['number_epochs'],
            gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'])
        gnn_time = time() - gnn_start


    elif args.task == 'mc':
        gnn_start = time()
        q_torch = gen_q_matrix_mc(args, nx_graph)

        net, optimizer, edges, edges_weight, inputs = get_gnn_mc(args, gnn_hypers, opt_params, args.TORCH_DEVICE,
                                                              args.TORCH_DTYPE, nx_graph)

        _, epoch, final_bitstring, best_bitstring, best_val, incumbents, timing = run_gnn_training_mc(args,
            q_torch, inputs, nx_graph, edges, edges_weight, net, optimizer, gnn_hypers['number_epochs'],
            gnn_hypers['tolerance'], gnn_hypers['patience'])
        gnn_time = time() - gnn_start


    if args.save:
        with open(result_path, 'w') as f:
            f.write('incumbents:' + str(incumbents) + '\n')
            f.write('timing:' + str(timing) + '\n')
            f.write('total time:' + str(gnn_time))