import os

import numpy as np
import torch
import argparse
import utils

from problem_parser import *
import time
from utils import *

import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='maxsat')
    parser.add_argument('--graph', type=str, default='reg')
    parser.add_argument('--Gset_id', type=int, default=66)
    parser.add_argument('--cnf_id', type=int, default=0)
    parser.add_argument('--cnf_k', type=int, default=3)
    parser.add_argument('--COLOR_name', type=str, default='anna')
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=8000)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--lr_x', type=float, default=0.02)
    parser.add_argument('--lr_y', type=float, default=0.02)
    parser.add_argument('--dual_init', type=float, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--TORCH_DTYPE', default=torch.float32)
    parser.add_argument('--TORCH_DEVICE', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--penalty', type=float, default=4)
    parser.add_argument('--solver', type=str, default='fem')
    parser.add_argument('--optimizer', choices=['gd', 'rmsprop', 'adam'], default='rmsprop',
                        help='Primal optimizer for --solver pdqubo (default: rmsprop)')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--timelimit', type=int, default=300)
    parser.add_argument('--num_boost', type=int, default=20)
    args = parser.parse_args()
    fix_seed(args.seed)

    if args.task != 'maxsat':
        RESULT_DIR = f'./result/{args.solver}/{args.task}/{args.graph}/'
    else:
        RESULT_DIR = f'./result/{args.solver}/maxsat/{args.cnf_k}CNF/'

    # if args.solver == 'pdqubo':
    #     RESULT_DIR = RESULT_DIR + f'lrx{args.lr_x}_lry{args.lr_y}_y{args.dual_init}_batch{args.batch}/'
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)


    if args.task == 'maxsat':
        graph = None
        RESULT_PATH = RESULT_DIR + f'{args.cnf_id}.txt'
    elif args.graph == 'reg':
        assert args.n is not None and args.d is not None
        RESULT_PATH = RESULT_DIR + f'n={args.n}d={args.d}s={args.seed}.txt'
        if os.path.exists(RESULT_PATH) and args.save:
            print("PASS", RESULT_PATH)
            sys.exit()
        graph = random_graph(n=args.n, d=args.d, seed=args.seed)

    elif args.graph == 'Gset':
        assert args.Gset_id is not None
        graph = parse_gset(f'{args.Gset_id}')
        RESULT_PATH = RESULT_DIR + f'G{args.Gset_id}.txt'
        if os.path.exists(RESULT_PATH) and args.save:
            print("PASS", RESULT_PATH)
            sys.exit()

    # elif args.graph == 'COLOR':
    #     assert args.COLOR_name is not None
    #     graph = parses_COLOR(f'{args.COLOR_name}')
    #     RESULT_DIR = RESULT_DIR + f'{args.COLOR_name}'
    # elif args.graph == 'bitcoin':
    #     graph = parse_btc()
    if args.task == 'mis':
        assert args.penalty is not None
        data = generate_MIS(graph, args.penalty)
    elif args.task == 'mc' or args.task == 'maxkcut':
        data = generate_Max_cut(graph)
    elif args.task == 'maxsat':
        names = os.listdir(f'./instance/MAXSAT/{args.cnf_k}CNF/')
        name = names[args.cnf_id]
        path = os.path.join(f'./instance/MAXSAT/{args.cnf_k}CNF/', name)
        data = generate_max_sat(path)

    if os.path.exists(RESULT_PATH) and args.save:
        print('PASS', RESULT_PATH)
        sys.exit()

    if args.solver == 'pdqubo':
        if args.k > 2 and args.task == 'maxkcut':
            from solver_jax import MAX_K_CUT_JAX
            solver = MAX_K_CUT_JAX(
                n_vars=data['num_vars'],
                Q_indices=data['Q_indices'],
                Q_values=data['Q_values'],
                c=data['c'],
                optimizer_type=args.optimizer,
                batch_size=args.batch,
                max_iters=args.max_iters,
                primal_lr=args.lr_x,
                dual_lr=args.lr_y,
                dual_init=args.dual_init,
                verbose=args.verbose,
                seed=args.seed,
            )
        elif args.task == 'maxsat':
            from solver_jax import MAXSAT_JAX
            solver = MAXSAT_JAX(
                n_vars=data['num_vars'],
                CNF=data['CNF'],
                optimizer_type=args.optimizer,
                batch_size=args.batch,
                max_iters=args.max_iters,
                primal_lr=args.lr_x,
                dual_lr=args.lr_y,
                dual_init=2,
                verbose=args.verbose,
            )
        else:
            from solver_jax import PDQUBO_JAX
            solver = PDQUBO_JAX(
                n_vars=data['num_vars'],
                Q_indices=data['Q_indices'],
                Q_values=data['Q_values'],
                c=data['c'],
                optimizer_type=args.optimizer,
                batch_size=args.batch,
                max_iters=args.max_iters,
                primal_lr=args.lr_x,
                dual_lr=args.lr_y,
                dual_init=args.dual_init,
                verbose=args.verbose,
                seed=args.seed,
            )


        t = time.perf_counter()
        solver.optimize()
        solving_time = time.perf_counter() - t


        if args.save:
            with open(RESULT_PATH, 'w') as f:
                f.write('incumbents:' + str(solver.objVal_record) + '\n')
                f.write('timing:' + str(solver.timing_record) + '\n')
                f.write('total time:' + str(solving_time) + '\n')
                # f.write('solution:')
                # for solx in solver.incumbent:
                #     f.write(str(int(solx)))

    elif args.solver == 'gurobi':
        from gurobi import gurobi_solve
        timing, incumbets = gurobi_solve(task=args.task, graph=graph)
        with open(RESULT_PATH, "w") as f:
            f.write(str(timing) + '\n')
            f.write(str(incumbets))
    elif args.solver == 'cra':
        from cra.cra import cra
        cra(args, graph, RESULT_PATH)
    elif args.solver == 'pignn':
        from pignn.pignn import pignn
        pignn(args, graph, RESULT_PATH)
    elif args.solver == 'fem':
        if args.task != 'maxsat':
            from fem.fem import fem
            fem(args, graph, RESULT_PATH)
        else:
            from fem.fem_maxsat import fem
            fem(path, RESULT_PATH)
    elif args.solver == 'anycsp':
        from anycsp.anycsp import ac
        ac(args, graph, RESULT_PATH)
    elif args.solver == 'ros':
        from ros.ros import ros
        data = ros(args, graph, RESULT_PATH)



