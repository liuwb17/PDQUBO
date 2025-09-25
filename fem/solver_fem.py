import torch
from .problem import OptimizationProblem
from math import log
import time


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
            dtype=torch.float32, seed=0, q=2, manual_grad=False,
            h_factor=0.01, sparse=False, args=None, graph=None
    ):
        self.dtype = dtype
        self.dev = dev
        self.args = args
        self.graph = graph
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
        self.binary = True if self.q == 2 else False
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def initialize(self):
        torch.manual_seed(0)
        if self.binary:
            h = self.h_factor * torch.randn(
                [self.num_trials, self.graph.number_of_nodes()],
                device=self.dev, dtype=self.dtype
            )

        else:
            h = self.h_factor * torch.randn(
                [self.num_trials, self.graph.number_of_nodes(), self.q],
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
        start_time = time.perf_counter()
        incumbents = [0.]
        timing = [0.]
        sol = None
        edges = torch.tensor([[u, v, w['weight']] for u, v, w in self.graph.edges(data=True)],
                             device=self.args.TORCH_DEVICE)
        edges_indices = edges[:, :2].long()
        edges_values = edges[:, 2]
        if self.args.task == 'mc':
            def obj(p):
                return (edges_values * (
                            p[:, edges_indices[:, 0]] + p[:, edges_indices[:, 1]] - 2 * p[:, edges_indices[:, 0]] * p[
                        :, edges_indices[:, 1]])).sum(-1)
        elif self.args.task == 'maxkcut':
            def obj(p):
                # the value of the cut
                # p [Batch, n, q]
                return (edges_values * (1 - (p[:, edges_indices[:, 0], :] * p[:, edges_indices[:, 1], :]).sum(-1))).sum(-1)
        elif self.args.task == 'mis':
            def obj(p):
                # the independent set size (0 if it is not independent)
                # p [Batch, n]

                return p.sum(-1) - self.args.penalty * (p[:, edges_indices[:, 0]] * p[:, edges_indices[:, 1]]).sum(-1)

        h = self.initialize()
        self.set_up_optimizer(h)
        for step in range(len(self.betas)):
            if time.perf_counter() - start_time >= self.args.timelimit:
                break
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
                free_energy =  - obj(p) - \
                              entropy(p) / self.betas[step]

                free_energy.backward(gradient=torch.ones_like(free_energy))  # minimize free energy
            self.opt.step()
            # print(free_energy.mean())
            # Incumbent callback
            if self.binary:
                int_p = torch.round(p).clone().detach()
            else:
                indices = torch.argmax(p.clone().detach(), dim=-1)  # 输出形状为 [b, n]
                int_p = torch.nn.functional.one_hot(indices, num_classes=p.size(2)).float()  # 输出形状为 [b, n, q]
            objs = obj(int_p)

            if self.args.task == 'mc' or self.args.task == 'maxkcut':
                cut, cut_idx = torch.max(objs, 0)
                if -cut < incumbents[-1]:
                    incumbents.append(-cut.item())
                    timing.append(time.perf_counter() - start_time)
                    sol = int_p[cut_idx]
            elif self.args.task == 'mis':
                ind_set, ind_set_idx = torch.max(objs, 0)
                assignment = int_p[ind_set_idx]
                vio = (assignment[edges_indices[:, 0]] * assignment[edges_indices[:, 1]]==1).any()
                if not vio and - ind_set < incumbents[-1]:
                    incumbents.append(- ind_set.item())
                    timing.append(time.perf_counter() - start_time)
                    sol = assignment
                print(step, (p**2-p).mean().item(), self.betas[step].item(), obj(p).mean().item())

        print(incumbents[-1])
        return sol, incumbents, timing

    def solve(self):
        sol, incumbents, timing = self.iterate()
        return sol, incumbents, timing
