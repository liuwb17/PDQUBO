import torch
import numpy as np
import sys
# sys.path.append("/data1/qyq/Benchmark-for-Max-k-Cut/anycsp")
from .src.csp.csp_data import CSP_Data
from .src.model.model import ANYCSP

from argparse import ArgumentParser
from .src.data.dataset import File_Dataset, File_Dataset_fromgraph




def ac(args, graph, result_path):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    name = 'best'
    if args.task == 'mc':
        model_dir = './anycsp/models/MAXCUT'
    elif args.task == 'maxkcut':
        model_dir = './anycsp/models/MAX3CUT'
    model = ANYCSP.load_model(args, model_dir, name)
    model.eval()
    model.to(device)

    dataset = File_Dataset_fromgraph(args, graph)

    num_solved = 0
    num_total = len(dataset)
    import time
    t0 = time.time()
    for data in dataset:
        max_val = data.constraints['ext'].cst_neg_mask.int().sum().cpu().numpy()
        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
        data.to(device)


        #with torch.cuda.amp.autocast():
        with torch.inference_mode():
            data, incumbents, timing = model(
                data,
                1000,
                return_all_assignments=True,
                return_log_probs=False,
                stop_early=True,
                verbose=args.verbose,
                keep_time=True,
                timeout=args.timelimit,
                graph=graph,
                args=args
            )
        best_per_run = data.best_num_unsat
        mean_best = best_per_run.mean()
        best = best_per_run.min().cpu().numpy()
        solved = best == 0
        num_solved += int(solved)
        best_cut_val = max_val - best
    print(f'Solved {100 * num_solved / num_total:.2f}%')
    print(time.time() - t0)
    print(data.best_assignment.cpu().numpy().shape)

    if args.save:
        with open(result_path, 'w') as f:
            f.write('incumbents:' + str(incumbents) + '\n')
            f.write('timing:' + str(timing) + '\n')
            f.write('total time:' + str(time.time() - t0))
    else:
        print('incumbents:' + str(incumbents) + '\n')
        print('timing:' + str(timing) + '\n')
        print('total time:' + str(time.time() - t0))