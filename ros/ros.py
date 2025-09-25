import time
from .utils import get_gnn_tuning, get_matrix, run_gnn_tuning, sample_one_hot
import torch


def ros(args, graph, result_path):
    args.n = graph.number_of_nodes()
    args.wd = 1e-4
    args.lr = 1e-2
    args.dim_embedding = 100
    args.hidden_dim = 100
    args.epochs = int(1e4)
    args.tol = 1e-2
    args.patience = 100
    args.max_iter = 100

    W = get_matrix(args, graph)
    incumbents = [0.]
    timing = [0.]
    gnn_start = time.time()
    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    print("Load gcn_model_ood_k" + str(args.k) + ".pth")
    net.load_state_dict(torch.load(f'./ros/model_data/gcn_model_ood_k{args.k}.pth', weights_only=True))
    best_solution_relaxed = run_gnn_tuning(args, W, edges, edges_weight, net, optimizer, args.epochs, args.tol, args.patience, inputs)
    Expectation = torch.trace(best_solution_relaxed @ W @ best_solution_relaxed.T)
    print("Expectation = " + str((W.sum() - Expectation).item()))
    best_val = torch.inf
    for _ in range(args.max_iter):
        Xt = sample_one_hot(best_solution_relaxed)
        val = torch.trace(Xt @ W @ Xt.T)
        # print(val, W.sum() - val)
        # incumbent_sol = Xt.argmax(axis=0)
        # different = (incumbent_sol[edge_indices[:, 0]] != incumbent_sol[edge_indices[:, 1]]).int()
        # cut = edge_values * different
        # print(val.item(), edge_values.sum().item() -cut.sum().item())
        if val < best_val:
            best_val = val
            best_solution = Xt
            incumbents.append((val - W.sum()).item())
            timing.append(time.time() - gnn_start)

    if args.save:
        with open(result_path, 'w') as f:
            f.write('incumbents:' + str(incumbents) + '\n')
            f.write('timing:' + str(timing) + '\n')
            f.write('total time:' + str(time.time() - gnn_start))

    
