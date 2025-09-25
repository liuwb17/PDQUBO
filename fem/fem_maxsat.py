import sys
import time

sys.path.append('')

import torch

num_trials = 1
num_steps = 1000
dev = 'cuda'
import torch, re
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import warnings

warnings.filterwarnings('ignore')


def parse_file(problem_type, filename, index_start=0):
    if problem_type in ['maxcut', 'bmincut', 'modularity', 'vertexcover']:
        n, m, couplings = read_graph(filename, index_start)
    elif problem_type == 'maxksat':
        n, m, couplings = read_cnf(filename)
    return n, m, couplings


def load_matrix(path: 'str', numer_package: 'str', store_format: 'str') -> 'float':
    """"
    Load the coupling matrix of the Graph instance from '.txt' file to python matrix.

    Parameters:

    :param path - The file path of the graph instance, with format of '.txt';
    :param numer_package - Choose the preferred python sci-package, choices = 'scipy' and 'torch';
    :param store_format - The output matrix will be with the store format of 'store_format', choices = 'csr', 'csc' and 'dense'.


    Returns:

    The output coupling matrix of the instance graph.

    """
    with open(path, "r") as f:
        l = f.readline()
        N, edges = [int(x) for x in l.split(" ") if x != "\n"]

    G = pd.read_csv(path, sep=' ', skiprows=[0], index_col=False, header=None, names=['node1', 'node2', 'weight'])
    G.fillna({'weight': int(1)}, inplace=True)
    shift = G.iloc[0, 0]
    ori_graph = np.array([list(np.concatenate([G.iloc[:, 0] - shift, G.iloc[:, 1] - shift])),
                          list(np.concatenate([G.iloc[:, 1] - shift, G.iloc[:, 0] - shift])),
                          list(np.concatenate([G.iloc[:, -1], G.iloc[:, -1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1, :], ori_graph[0, :])).tolist()].T
    if numer_package == 'scipy':
        J = coo_matrix((ori_graph[2, :].tolist(),
                        (ori_graph[0, :].tolist(), ori_graph[1, :].tolist())), shape=(N, N))
        if J.shape[0] != N:
            print("The shape of J does not match N!")
        if J.data.shape[0] / 2 != edges:
            print("The number of elements in J does not match edges!")
        if store_format == 'csr':
            J = csr_matrix(J)
        elif store_format == 'csc':
            J = csc_matrix(J)
        elif store_format == 'dense':
            J = J.todense()
        else:
            print("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    elif numer_package == 'torch':
        J = torch.sparse_coo_tensor([ori_graph[0, :].tolist(),
                                     ori_graph[1, :].tolist()],
                                    ori_graph[2, :].tolist(), (N, N))
        if J.shape[0] != N:
            print("The shape of J does not match N!")
        if J._values().shape[0] / 2 != edges:
            print("The number of elements in J does not match edges!")

        if store_format == 'csr':
            J = J.to_sparse_csr()
        elif store_format == 'csc':
            J = J.to_sparse_csc()
        elif store_format == 'dense':
            J = J.to_dense()
        else:
            print("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    else:
        print("Error: Input wrong 'numer_package'! Please choose from ['scipy', 'torch'].")
    return J


def load_gset(instance):
    """
    load the weight matrix of Gset, modified from code of Zisong Shen
    """
    # print('loading Gset',instance,'...')
    path = './Gset/' + instance
    G = pd.read_csv(path, sep=' ')
    n_v = int(G.columns[0])
    ori_graph = np.array([list(np.concatenate([G.iloc[:, 0] - 1, G.iloc[:, 1] - 1])),
                          list(np.concatenate([G.iloc[:, 1] - 1, G.iloc[:, 0] - 1])),
                          list(np.concatenate([G.iloc[:, -1], G.iloc[:, -1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1, :], ori_graph[0, :])).tolist()].T
    J = torch.sparse_coo_tensor([ori_graph[0, :].tolist(), ori_graph[1, :].tolist()], ori_graph[2, :].tolist(),
                                (n_v, n_v)).to_sparse_csr()
    """ using sparse column here """
    with open('targetvalue.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    result = re.findall(".*" + instance + " (.*).*", content)
    target_value = int(result[0]) if result else 0
    # print('N=%d'%(J.shape[0])," c=%.2f"%(torch.count_nonzero(J.to_dense())*2/J.shape[0]),"best_cut:"+ target_value)
    return J.to_dense(), target_value


def read_graph(file, index_start=0):
    """
    function for reading graph files
    the specific format should be n m in the first line, and m following lines
    represent source end weight
    Parameters:
        file: string, the filenmae of the graph to be readed
        index_start: int, specify which is the start index of the graph
    """
    with open(file, "r") as f:
        l = f.readline()
        n, m = [int(x) for x in l.split(" ") if x != "\n"]
        J = torch.zeros([n, n])
        neighbors = [[] for i in range(n)]
        for k in range(m):
            l = f.readline()
            l_split = l.split()
            i, j = [int(x) for x in l_split[:2]]
            if len(l_split) == 2:
                w = 1.0
            elif len(l_split) == 3:
                w = float(l_split[2])
            else:
                raise ValueError("Unkown graph file format")
            i -= index_start
            j -= index_start
            J[i, j], J[j, i] = w, w
            neighbors[i].append(j)
            neighbors[j].append(i)
    return n, m, J


def read_cnf(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    k_length_sat_table = {}
    for line in lines:
        l = line.split()
        if l[0] == 'c':  # comment line
            pass
        elif l[0] == 'p':  # problem line
            N, M = map(int, l[2:])
        else:
            clause = list(map(int, l[:-1]))
            k = len(clause)
            if k not in k_length_sat_table:
                k_length_sat_table[k] = []
            k_length_sat_table[k].append([])
            k_length_sat_table[k][-1].append(list(map(abs, clause)))
            q_states = []
            for i in range(k):
                if clause[i] > 0:
                    q_states.append(0)  # postive literal
                else:
                    q_states.append(1)  # negative literal
            k_length_sat_table[k][-1].append(q_states)
    sat_table = []
    minimum_index = []
    maximum_index = []
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key] = np.array(k_length_sat_table[key])
        minimum_index.append(np.min(k_length_sat_table[key][:, 0, :]))
        maximum_index.append(np.max(k_length_sat_table[key][:, 0, :]))
    max_idx = max(maximum_index)
    min_idx = min(minimum_index)
    assert max_idx - min_idx + 1 == N
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key][:, 0, :] -= min_idx
        k_length_sat_table[key] = k_length_sat_table[key].tolist()
        sat_table += k_length_sat_table[key]
    real_M = len(sat_table)
    assert real_M == M
    max_k = max(k_length_sat_table.keys())
    min_k = min(k_length_sat_table.keys())
    if max_k != min_k:
        raise ValueError("This is not a max-ksat instances.")
    mask_tensor = clause_mask_tensor(N, M, sat_table)
    return N, M, mask_tensor


def clause_mask_tensor(N, M, sat_table):
    clause = []
    for ii in range(M):
        k = len(sat_table[ii][0])
        clause_m = torch.sparse_coo_tensor(
            [sat_table[ii][0], sat_table[ii][1]],
            [1] * k,
            (N, 2)
        ).to_dense().unsqueeze(0)
        clause.append(clause_m.unsqueeze(0))
    clause_batch = torch.cat(clause, dim=0)  # [M, batch, N, q]  # sparse tensor values = k * M * batch
    return clause_batch


import torch
import torch.nn.functional as Func


def infer_qubo(J, p):
    """
    determine the configuration by mariginal probility and calculate the value
    of QUBO function
    """
    config = p.round()
    return config, expected_qubo(J, config)


def expected_qubo(J, p):
    """
    QUBO function of weights and mariginal probability p
    Parameters:
        J: torch.Tensor, shape: (N, N), weight matrix of the QUBO problem
        p: torch.Tensor, shape: (batch, N), mariginal probability of QUBO variables
    """
    return torch.bmm(
        (p @ J).reshape(-1, 1, J.shape[1]),
        p.reshape(-1, p.shape[1], 1)
    ).reshape(-1)


def manual_grad_qubo(J, p):
    """
    gradients of QUBO function
    """
    grad = 2 * (p * (1 - p) * p @ J)
    # grad = 2 * (p*(1-p) * (p > 0.5).to(J.dtype) @ J)
    return grad


def infer_bmincut(J, p):
    """
    J: weight matrix, with shape [N,N], better with the csc format
    p: the marginal matrix, with shape [batch, N, q], q is the number of groups
    config is the configuration for n variables, with shape [batch, N, q]
    return the cut size, i.e. outer weights.
    """
    config = Func.one_hot(p.view(-1, J.shape[0], p.shape[-1]).argmax(dim=2), num_classes=p.shape[-1]).to(J.dtype)
    return config, expected_bmincut(J, config) / 2


def expected_bmincut(J, p):
    """
    p is the marginal matrix, with shape [batch, N,q], q is the number of groups
    config is the configuration for n variables, with shape [batch, N,q]
    return TWICE the expected cut size, i.e. outer weights.
    """
    return ((J @ p) * (1 - p)).sum((1, 2))


def manual_grad_bmincut(J, p, imba):
    temp = 1 - 2 * p
    tp = J @ temp + imba * (2 * p.sum(1, keepdim=True) - 2 * p)
    h_grad = (tp - (tp * p).sum(2, keepdim=True).expand(tp.shape)) * p
    return h_grad


def infer_maxcut(J, p):
    """
    J: weight matrix, with shape [N, N]
    p: the marginal matrix, with shape [batch, N], p[:, x] represent the
        probability of x variable to be 1
    config is the configuration for N variables, with shape [batch, N]
    return the cut size, i.e. outer weights.
    """
    config = p.round()
    return config, expected_cut(J, config) / 2


def expected_cut(J, p):
    """
    p is the marginal matrix, with shape [batch, N]
    config is the configuration for n variables, with shape [batch, N]
    return TWICE the expected cut size, i.e. outer weights.
    """
    return 2 * ((p @ J) * (1 - p)).sum(1)


# def expected_maxcut(J, p):
#     return -torch.bmm(
#         (p @ J).reshape(-1, 1, J.shape[1]),
#         p.reshape(-1, p.shape[1], 1)
#     ).reshape(-1)


# def manual_grad_maxcut(J, p):
#     temp = 1 - 2 * p
#     tp = -J @ temp
#     h_grad = (tp  - (tp * p).sum(2,keepdim=True).expand(tp.shape))*p
#     return h_grad
def manual_grad_maxcut(J, p, discretization=False):
    p_prime = p.round() if discretization else p
    h_grad = (2 * p_prime - 1) @ J * (1 - p) * p
    return h_grad


def manual_grad_modularity(J, p, m, d):
    temp = d * p
    tp = -J @ p + d * ((temp).sum(1, keepdim=True) - temp) / m
    h_grad = (tp - (tp * p).sum(2, keepdim=True).expand(tp.shape)) * p
    return h_grad


def imbalance_penalty(p):
    """
    p is the marginal matrix, with shape [batch, N,q], q is the number of groups
    config is the configuration for n variables, with shape [batch, N,q]
    return an anti-ferromagnetic all-to-all interaction panelty which equals to
    #   \sum_i\sum_{s_i}\sum_{j\neq i}p_i(s_i)p_j(s_i)
    # = \sum_{s_i}\sum_ip_i(s_i)\sum_jp_j(s_i) - \sum_i\sum_{s_i}p_i(s_i)*p_i(s_i)
    # = \sum_{s_i}(\sum_{i}p_i(s_i))**2 - \sum_i\sum_{s_i}p_i(s_i)*p_i(s_i)

    """
    return ((p.sum(1)) ** 2).sum(1) - (p * p).sum(2).sum(1)


def expected_inner_weight(J, p):
    return 0.5 * (J.sum() - expected_cut(J, p))


def expected_inner_weight_configmodel(J, p):
    """
    \frac{1}{2m}\sum_i\sum_j\frac{d_i*d_j}\delta(s_i,s_j)
    =\frac{1}{2m}\sum_i\sum_{s_i}d_i\sum_{j\neq i}d_jp_i(s_i)p_j(s_i)
    =\frac{1}{2m}\sum_i\sum_{s_i}d_ip_i(s_i)\sum_{j\neq i}d_jp_j(s_i)
    =\frac{1}{2m}(\sum_{s_i}\sum_id_ip_i(s_i)\sum_{j}d_jp_j(s_i) - \sum_id_id_i\sum_{s_i}p_i(s_i)*p_i(s_i))
    =\frac{1}{2m}(\sum_{s_i}(\sum_id_ip_i(s_i))**2 - \sum_i\sum_{s_i}d_i*d_i*p_i(s_i)*p_i(s_i))
    """
    d = J.to_dense().sum(1).reshape([1, p.shape[1], 1]).expand(p.shape)
    m2 = J.sum()
    return (((d * p).sum(1) ** 2).sum(1) - (d * p * d * p).sum(2).sum(1)) / m2


def manual_grad_maxksat(clause_batch, p):
    M, batch = clause_batch.shape[:2]
    minus_p = 1 - 0.99999 * p
    prod = clause_batch * minus_p
    value = prod._values().reshape(M, batch, -1)  # # values = k * M * batch
    value_prod = prod._values().reshape(M, batch, -1).prod(-1, keepdim=True)
    grad = torch.sparse_coo_tensor(
        prod.coalesce().indices(),
        (-value_prod / value).reshape([1, -1]).squeeze(0),
        prod.shape
    ).sum(0, keepdim=True).to_dense()
    h_grad = (grad - (grad * p).sum(3, keepdim=True).expand(grad.shape)) * p
    return h_grad


def expected_maxksat(clause_batch, p):
    M, batch = clause_batch.shape[:2]
    minus_p = 1 - p
    prod = clause_batch * minus_p
    value_prod = prod._values().reshape(M, batch, -1).prod(-1, keepdim=True)
    energy = value_prod.sum(0).reshape(1, -1).squeeze(0)
    return energy


def infer_maxksat(clause_batch, p):
    config = Func.one_hot(p.view(1, clause_batch.shape[1], clause_batch.shape[2], -1).argmax(dim=3),
                          num_classes=p.shape[-1]).to(clause_batch.dtype)
    return config, expected_maxksat(clause_batch, config)


class OptimizationProblem:
    """
    Optimization problem class
    """

    def __init__(
            self,
            num_nodes, num_interactions,
            coupling_matrix,
            problem_type,
            imbalance_weight=5.0,
            discretization=False,
            customize_expected_func=None,
            customize_grad_func=None,
            customize_infer_func=None
    ) -> None:
        self.num_nodes = num_nodes
        self.num_interactions = num_interactions
        self.coupling_matrix = coupling_matrix
        self.problem_type = problem_type
        self.imbalance_weight = imbalance_weight
        self.discretization = discretization
        self.constant = 0
        self.customize_expected_func = customize_expected_func
        self.customize_grad_func = customize_grad_func
        self.customize_infer_func = customize_infer_func
        pass

    def extra_preparation(self, num_trials=1, sparse=False):
        if self.problem_type == 'maxcut':
            self.c = 1 / torch.abs(self.coupling_matrix).sum(1)
        if self.problem_type == 'bmincut':
            self.w2 = self.coupling_matrix.square().sum()
            self.imbalance_weight = self.imbalance_weight * self.w2 / (self.num_nodes ** 2)
        if self.problem_type == 'modularity':
            self.d = self.coupling_matrix.sum(1).reshape([1, self.num_nodes, 1])
            self.m = self.coupling_matrix.sum() / 2
        if self.problem_type == 'vertexcover':
            degrees = self.coupling_matrix.sum(1)
            self.coupling_matrix *= self.imbalance_weight / 2
            self.coupling_matrix[range(self.num_nodes), range(self.num_nodes)] = \
                1 - degrees * self.imbalance_weight
            self.constant = self.num_interactions * self.imbalance_weight
        if self.problem_type == 'maxksat':
            self.coupling_matrix = self.coupling_matrix.repeat(1, num_trials, 1, 1)
        if sparse:
            self.coupling_matrix = self.coupling_matrix.to_sparse()

    def set_up_couplings_status(self, dev, dtype):
        self.coupling_matrix = self.coupling_matrix.to(dtype).to(dev)

    def expectation(self, p):
        if self.problem_type == 'maxcut':
            return -expected_cut(self.coupling_matrix / 2, p)
        elif self.problem_type == 'bmincut':
            return expected_bmincut(self.coupling_matrix, p) + \
                self.imbalance_weight * imbalance_penalty(p)
        elif self.problem_type == 'modularity':
            return -expected_inner_weight(self.coupling_matrix, p) + \
                expected_inner_weight_configmodel(self.coupling_matrix, p)
        elif self.problem_type == 'vertexcover':
            return expected_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            return expected_maxksat(self.coupling_matrix, p.unsqueeze(0))
        elif self.problem_type == 'customize':
            return self.customize_expected_func(self.coupling_matrix, p)

    def manual_grad(self, p):
        if self.problem_type == 'maxcut':
            return manual_grad_maxcut(self.c * self.coupling_matrix, p, self.discretization)
        elif self.problem_type == 'bmincut':
            return manual_grad_bmincut(self.coupling_matrix, p, self.imbalance_weight)
        elif self.problem_type == 'modularity':
            return manual_grad_modularity(
                self.coupling_matrix, p, self.m, self.d.expand(p.shape)
            )
        elif self.problem_type == 'vertexcover':
            return manual_grad_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            return manual_grad_maxksat(self.coupling_matrix, p.unsqueeze(0)).squeeze(0)
        elif self.problem_type == 'customize':
            return self.customize_grad_func(self.coupling_matrix, p)

    def inference_value(self, p):
        p = torch.vstack([pi for pi in p if torch.isnan(pi).sum() == 0])
        if self.problem_type == 'maxcut':
            config, result = infer_maxcut(self.coupling_matrix, p)
        elif self.problem_type == 'bmincut':
            config, result = infer_bmincut(self.coupling_matrix, p)
        elif self.problem_type == 'vertexcover':
            config, result = infer_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            config, result = infer_maxksat(self.coupling_matrix, p.unsqueeze(0))
        elif self.problem_type == 'customize':
            return self.customize_infer_func(self.coupling_matrix, p)
        result += self.constant
        return config, result


import torch
from math import log


def entropy_q(p):
    """
    p is the probabilities for each group, shape [batch, N, q], with q denoting the number of groups
    return - \sum_{i=1}^N sum_{t=1}^q p(t)*\log p(t)
    """
    return - (p * torch.log(p)).sum(2).sum(1)


def entropy_grad_q(p):
    return -p * (torch.log(p) - (p * torch.log(p)).sum(2, keepdim=True).expand(p.shape))


def entropy_binary(p):
    return - ((p * torch.log(p)) + (1 - p) * torch.log(1 - p)).sum(1)


def entropy_grad_binary(p):
    grad = - (p * (1 - p) * (p.log() - (1 - p).log()))
    return grad


class Solver:
    def __init__(
            self,
            problem, num_trials, num_steps, betamin=0.01, betamax=0.5,
            anneal='inverse', optimizer='adam', learning_rate=0.1, dev='cuda',
            dtype=torch.float32, seed=1, q=2, manual_grad=False,
            h_factor=0.01, sparse=False
    ):
        self.dtype = dtype
        self.dev = dev
        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax), num_steps))
        elif anneal == 'inverse':
            betas = 1 / torch.linspace(betamax, betamin, num_steps)
        self.betas = betas.to(self.dtype).to(self.dev)
        self.num_trials = num_trials
        self.seed = seed
        self.q = q
        self.manual_grad = manual_grad
        self.h_factor = h_factor
        self.problem = problem
        self.problem.set_up_couplings_status(dev, dtype)
        self.problem.extra_preparation(num_trials, sparse)
        self.binary = True if self.problem.problem_type in ['maxcut', 'vertexcover'] else False
        if self.binary:
            assert self.q == 2
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def initialize(self):
        torch.manual_seed(self.seed)
        if self.binary:
            h = self.h_factor * torch.randn(
                [self.num_trials, self.problem.num_nodes],
                device=self.dev, dtype=self.dtype
            )
        else:
            h = self.h_factor * torch.randn(
                [self.num_trials, self.problem.num_nodes, self.q],
                device=self.dev, dtype=self.dtype
            )
        if self.manual_grad:
            h.requires_grad = False
        else:
            h.requires_grad = True
        return h

    def set_up_optimizer(self, params):
        if self.optimizer == 'adam':
            self.opt = torch.optim.Adam([params], lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.opt = torch.optim.RMSprop(
                [params], lr=self.learning_rate, alpha=0.98, eps=1e-08,
                weight_decay=0.01, momentum=0.91, centered=False
            )
        else:
            raise ValueError("Unkown optimizer, valid choices are ['adam', 'rmsprop'].")

    def iterate(self):
        h = self.initialize()
        self.set_up_optimizer(h)
        for step in range(len(self.betas)):
            p = torch.sigmoid(h) if self.binary else torch.softmax(h, dim=2)
            self.opt.zero_grad()
            if self.binary:
                entropy_grad = entropy_grad_binary
                entropy = entropy_binary
            else:
                entropy_grad = entropy_grad_q
                entropy = entropy_q
            if self.manual_grad:
                h.grad = self.problem.manual_grad(p) - \
                         entropy_grad(p) / self.betas[step]
            else:
                free_energy = self.problem.expectation(p) - \
                              entropy(p) / self.betas[step]
                free_energy.backward(gradient=torch.ones_like(free_energy))  # minimize free energy
            self.opt.step()
        return p

    def solve(self):
        marginal = self.iterate()
        configs, results = self.problem.inference_value(marginal)
        return configs, results

class FEM:
    """
    Interface class of the solver
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_file(cls, problem_type, filename, index_start=0, **args):
        num_nodes, num_interactions, couplings = parse_file(
            problem_type, filename, index_start
        )
        fem = cls()
        fem.set_up_problem(
            num_nodes, num_interactions, problem_type, couplings, **args
        )
        return fem

    @classmethod
    def from_couplings(cls, problem_type, num_nodes, num_interactions, couplings, **args):
        fem = cls()
        fem.set_up_problem(
            num_nodes, num_interactions, problem_type, couplings, **args
        )
        return fem

    def set_up_problem(
            self, num_nodes, num_interactions, problem_type, coupling_matrix,
            imbalance_weight=5.0, discretization=False,
            customize_expected_func=None, customize_grad_func=None,
            customize_infer_func=None
    ):
        supported_types = [
            'maxcut', 'bmincut', 'modularity', 'maxksat', 'vertexcover', 'customize'
        ]
        if problem_type not in supported_types:
            raise ValueError(
                f"Problem type '{problem_type}', current support types are {supported_types}"
            )
        self.problem = OptimizationProblem(
            num_nodes, num_interactions, coupling_matrix,
            problem_type, imbalance_weight, discretization,
            customize_expected_func, customize_grad_func, customize_infer_func
        )

    def set_up_solver(
            self, num_trials, num_steps, betamin=0.01, betamax=0.5,
            anneal='inverse', optimizer='adam', learning_rate=0.1, dev='cuda',
            dtype=torch.float32, seed=1, q=2, manual_grad=False,
            h_factor=0.01, sparse=False
    ):
        assert 'problem' in self.__dict__.keys()
        self.solver = Solver(
            self.problem, num_trials, num_steps, betamin, betamax, anneal,
            optimizer, learning_rate, dev, dtype, seed, q, manual_grad,
            h_factor, sparse
        )

    def solve(self):
        return self.solver.solve()


def fem(file, result_path):
    case_maxksat = FEM.from_file(
        'maxksat', file
    )
    case_maxksat.set_up_solver(
        num_trials, num_steps, manual_grad=True, h_factor=0.3, anneal='lin',
        betamin=0.01, betamax=30, learning_rate=1.1, sparse=True, dev=dev,
    )
    t = time.perf_counter()
    config, result = case_maxksat.solve()
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    with open(result_path, 'w') as f:
        f.write(f'maxksat test instance, optimal value:{result.min()}' + '\n')
        f.write(f'time:{time.perf_counter() - t}')