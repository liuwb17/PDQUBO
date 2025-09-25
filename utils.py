import torch
import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def grb_solve(data, k=None):
    n = data['num_vars']
    m = gp.Model()


    if k is None:
        x = m.addVars(n, vtype=gp.GRB.BINARY, name="x")
        m.update()
        obj = gp.LinExpr()

        for id, indice in enumerate(data['Q_indices'].T):
            i, j = indice[0], indice[1]
            obj += data['Q_values'][id] * x[i] * x[j]
        for i in range(n):
            obj += data['c'][i] * x[i]

        m.setObjective(obj, gp.GRB.MINIMIZE)
        m.Params.MIPFocus = 1

        m.optimize()
    else:
        Q = data['Q_sparse'].todense()
        indices = data['Q_indices'].T
        x = m.addVars(n, k, vtype=GRB.BINARY, name="x")

        obj = gp.quicksum(0.5 * Q[i, j] * (1 - gp.quicksum(x[i, c] * x[j, c] for c in range(k))) for (i, j) in indices)
        m.setObjective(obj, GRB.MAXIMIZE)

        for i in range(n):
            m.addConstr(gp.quicksum(x[i, c] for c in range(k)) == 1)

        m.optimize()

    return m