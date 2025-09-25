import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GraphConv
from itertools import chain, islice
import time
import numpy as np
from itertools import chain, islice, combinations
from collections import OrderedDict, defaultdict
import dgl

# GNN class to be instantiated with specified param values
class GCN_dev_mc(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, device):
        super(GCN_dev_mc, self).__init__()

        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.bn1 = GraphNorm(hidden_size)
        self.conv2 = GraphConv(hidden_size, number_classes).to(device)
        self.bn2 = GraphNorm(number_classes)

    def forward(self, inputs, edges, edges_weight):
        h = self.conv1(inputs, edges, edges_weight)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.conv2(h, edges, edges_weight)
        h = self.bn2(h)
        h = torch.sigmoid(h)
        return h

class GCN_dev_mis(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device):
        super(GCN_dev_mis, self).__init__()
        self.dropout_frac = dropout
        self.conv1 = dgl.nn.pytorch.GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = dgl.nn.pytorch.GraphConv(hidden_size, number_classes).to(device)

    def forward(self, g, inputs):
        # input step
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(g, h)
        h = torch.sigmoid(h)

        return h

# Generate random graph of specified size and type,
# with specified degree (d) or edge probability (p)
def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph


# helper function to convert Q dictionary to torch tensor
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    n_nodes = len(nx_G.nodes)
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)
    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat


# Chunk long list
def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


# helper function for custom loss according to Q matrix
def loss_func(probs, Q_mat):
    cost = probs.T @ Q_mat @ probs

    return cost


# Construct graph to learn on
def get_gnn_mc(args, gnn_hypers, opt_params, torch_device, torch_dtype, graph):
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    number_classes = gnn_hypers['number_classes']

    # inputs = torch.rand((args.n, dim_embedding)).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)

    inputs = nn.Embedding(graph.number_of_nodes(), dim_embedding).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)

    net = GCN_dev_mc(dim_embedding, hidden_dim, number_classes, torch_device)

    net = net.type(torch_dtype).to(torch_device)
    optimizer = torch.optim.Adam(net.parameters(), **opt_params)

    edges = []
    edges_weight = []
    for (u, v, w) in graph.edges(data=True):
        val = w['weight']
        edges.append([u, v])
        edges.append([v, u])
        edges_weight.append(val)
        edges_weight.append(val)
    edges = torch.tensor(edges).transpose(1, 0).to(torch_device)
    edges_weight = torch.tensor(edges_weight).type(torch_dtype).to(torch_device)
    return net, optimizer, edges, edges_weight, inputs


def get_gnn_mis(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):

    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']

    # instantiate the GNN
    net = GCN_dev_mis(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, embed, optimizer

def gen_q_matrix_mis(args, nx_G):
    n = nx_G.number_of_nodes()
    Q = torch.zeros((n, n)).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    for i in nx_G.nodes:
        Q[i][i] = - 1
    for (u, v) in nx_G.edges:
        Q[u][v] = args.penalty / 2
        Q[v][u] = args.penalty / 2
    return Q


def gen_q_matrix_mc(args, nx_G):
    n = nx_G.number_of_nodes()
    Q = torch.zeros((n, n)).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)

    for (u, v, w) in nx_G.edges(data=True):
        assert u != w
        Q[u][v] = w['weight']
        Q[v][u] = w['weight']
        Q[u][u] -= w['weight']
        Q[v][v] -= w['weight']
    return Q


# Calculate results given bitstring and graph definition, includes check for violations
def postprocess_gnn_mis(best_bitstring, nx_graph):
    # get bitstring as list
    bitstring_list = list(best_bitstring)

    # compute cost
    size_mis = sum(bitstring_list)

    # get independent set
    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))

    print('Calculating violations...')
    # check for violations
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return size_mis, ind_set, number_violations


def run_gnn_training_mc(args, q_torch, inputs, graph, edges, edges_weight, net, optimizer, number_epochs, tol, patience):

    edges_indices = torch.tensor([[u, v, w['weight']] for u, v, w in graph.edges(data=True)]).cuda().long()
    edges_val = edges_indices[:, 2].float()

    prev_loss = 1.
    count = 0
    best_loss = np.inf
    incumbents = [0.]
    timing = [0.]
    t_gnn_start = time.time()

    for epoch in range(number_epochs):
        if time.time() - t_gnn_start >= args.timelimit:
            break
        probs = net(inputs.weight, edges, edges_weight)

        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        bitstring = (probs.detach() >= 0.5) * 1
        bitstring = bitstring.squeeze(1)
        if loss_ < best_loss:
            best_loss = loss_
            best_bitstring = bitstring

        cut = (edges_val * (bitstring[edges_indices[:, 0]] + bitstring[edges_indices[:, 1]] - 2 * bitstring[
            edges_indices[:, 0]] * bitstring[edges_indices[:, 1]])).sum()

        if -cut < incumbents[-1]:
            incumbents.append(-cut.item())
            timing.append(time.time() - t_gnn_start)

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')

        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f"stopping early on epoch {epoch} (patience: {patience})")
            break

        prev_loss = loss_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_bitstring = (probs.detach() >= 0.5) * 1
    best_val = loss_func(best_bitstring.float(), q_torch)
    return net, epoch, final_bitstring, best_bitstring, best_val.item(), incumbents, timing


def run_gnn_training_mis(args, graph, q_torch, dgl_graph, net, embed, optimizer, number_epochs, tol, patience, prob_threshold):
    edges_indices = torch.tensor([[u, v] for u, v in graph.edges()]).cuda().long()
    # Assign variable for user reference
    inputs = embed.weight

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered

    # initialize optimal solution
    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(q_torch.dtype).to(q_torch.device)
    best_loss = loss_func(best_bitstring.float(), q_torch)

    t_gnn_start = time.time()
    incumbents = [0.]
    timing = [0.]
    # Training logic
    for epoch in range(number_epochs):
        print('spent time=', time.time() - t_gnn_start, 'time limit=', args.timelimit)
        if time.time() - t_gnn_start >= args.timelimit:
            break
        # get logits/activations
        probs = net(dgl_graph, inputs)[:, 0]  # collapse extra dimension output from model

        # build cost value with QUBO cost function
        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        # Apply projection
        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring
        set = bitstring.sum()
        vio = (bitstring[
                   edges_indices[:, 0]] * bitstring[edges_indices[:, 1]]).sum()
        if vio == 0 and -set < incumbents[-1]:
            incumbents.append(-set.item())
            timing.append(time.time() - t_gnn_start)
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')

        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break

        # update loss tracking
        prev_loss = loss_

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()        # calculate gradient through compute graph
        optimizer.step()       # take step, update weights

    t_gnn = time.time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_loss}')

    final_bitstring = (probs.detach() >= prob_threshold) * 1

    return net, epoch, final_bitstring, best_bitstring, incumbents, timing