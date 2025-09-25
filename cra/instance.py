
from collections import OrderedDict, defaultdict

def gen_q_dict_mis_sym(nx_G, penalty=4):
    Q_dic = defaultdict(int)
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = int(penalty / 2)
        Q_dic[(v, u)] = int(penalty / 2)
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -1
    return Q_dic


def gen_q_dict_max_cut_sym(nx_G):
    Q_dic = defaultdict(int)
    for u, v, w in nx_G.edges(data=True):
        w = w['weight']
        Q_dic[(u,u)]+= -w
        Q_dic[(v,v)]+= -w
        Q_dic[(u,v)]+= w
        Q_dic[(v,u)]+= w
    return Q_dic
