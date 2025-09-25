import torch
from .interface import FEM
import time
def fem(args, graph, result_path):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(
        precision=6,
        sci_mode=False
    )
    num_trials = args.batch
    num_steps = 1000
    dev = 'cuda'

    if args.task == 'mc':
        t = time.perf_counter()
        case_maxcut = FEM.from_file(
            'maxcut', f'./instance/Gset/G{args.Gset_id}.txt', index_start=1, discretization=True
        )

        case_maxcut.set_up_solver(
            num_trials, num_steps, manual_grad=True, betamin=0.001, betamax=0.5,
            learning_rate=0.1, optimizer='rmsprop', dev=dev, args=args, graph=graph
        )

        sol, incumbents, timing = case_maxcut.solve()
        total_time = time.perf_counter() - t
        # edges = torch.tensor([[u, v, w['weight']] for u, v, w in graph.edges(data=True)],
        #                      device=args.TORCH_DEVICE)
        # edges_indices = edges[:, :2].long()
        # edges_values = edges[:, 2]
        # cut = edges_values * (sol[edges_indices[:, 0]]!=sol[edges_indices[:, 1]]).float()
        # assert cut.sum().int().cpu() + int(incumbents[-1]) == 0


    elif args.task == 'maxkcut':
        t = time.perf_counter()
        case_maxcut = FEM.from_file(
            'maxcut', f'./instance/Gset/G{args.Gset_id}.txt', index_start=1, discretization=True
        )

        case_maxcut.set_up_solver(
            num_trials, num_steps, manual_grad=False, betamin=0.001, betamax=0.5, q=3,
            learning_rate=0.1, optimizer='rmsprop', dev=dev, args=args, graph=graph
        )

        sol, incumbents, timing = case_maxcut.solve()

        total_time = time.perf_counter() - t
        # sol_flatten = sol.argmax(-1)
        # edges = torch.tensor([[u, v, w['weight']] for u, v, w in graph.edges(data=True)],
        #                      device=args.TORCH_DEVICE)
        # edges_indices = edges[:, :2].long()
        # edges_values = edges[:, 2]
        #
        # cut = edges_values * (sol_flatten[edges_indices[:, 0]]!=sol_flatten[edges_indices[:, 1]]).float()
        # assert cut.sum().int().cpu().numpy() + int(incumbents[-1]) == 0

    elif args.task == 'mis':
        t = time.perf_counter()
        case_maxcut = FEM.from_file(
            'maxcut', f'./instance/Gset/G{args.Gset_id}.txt', index_start=1, discretization=False
        )

        case_maxcut.set_up_solver(
            num_trials, 1000, manual_grad=False, betamin=0.001, betamax=4,
            learning_rate=0.1, optimizer='rmsprop', dev=dev, args=args, graph=graph
        )

        sol, incumbents, timing = case_maxcut.solve()
        total_time = time.perf_counter() - t

    elif args.task == 'maxsat':
        t = time.perf_counter()
        case_maxksat = FEM.from_file(
            'maxksat', '/data1/lwb/FEM/benchmarks/maxsat/3CNF/00_10000v_75000c.cnf'
        )
        case_maxksat.set_up_solver(
            num_trials, num_steps, manual_grad=True, h_factor=0.3, anneal='lin',
            betamin=0.01, betamax=30, learning_rate=1.1, sparse=True, dev=dev, args=args,q=2
        )
        config, result = case_maxksat.solve()
        if args.save:
            with open(result_path, 'w') as f:
                f.write(f'maxksat optimal value:{result.min()}')
                f.write(f'time:{time.perf_counter() - t}')
            quit()

    if args.save:
        with open(result_path, 'w') as f:
            f.write('incumbents:' + str(incumbents) + '\n')
            f.write('timing:' + str(timing) + '\n')
            f.write('total time:' + str(total_time))