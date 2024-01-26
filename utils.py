import os
import numpy as np
import torch
import torch.nn.functional as F

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = np.sum(count) / count[i]

    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)

    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    if dist.is_cuda:
        dist = dist.reshape(-1,).cpu().numpy()
    else:
        dist = dist.reshape(-1, ).numpy()
    dist.sort()
    parameter = dist[edge_per_node*data.shape[0]]

    return parameter


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist < parameter).float().numpy()
    if self_dist:
        np.fill_diagonal(g, 0)
        g = torch.tensor(g)
    return g


def gen_adj_mat_tensor(data, parameter, i, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    mask = adj_T > adj
    adj[mask] = adj[mask]+adj_T[mask]-adj[mask]
    adj = F.normalize(adj + I, p=1)
    print('# of edge in training similarity network for data type ' + str(i) + ': ' + str(torch.sum(adj > 0)))
    adj = to_sparse(adj)

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, i, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    num_tr = len(trte_idx["tr"])

    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr,num_tr:] = 1-dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te

    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr # retain selected edges

    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])

    mask = adj_T > adj
    adj[mask] = adj[mask]+adj_T[mask]-adj[mask]
    adj = F.normalize(adj + I, p=1)
    print('# of edge in testing similarity network for data type ' + str(i) + ': ' + str(torch.sum(adj > 0)))
    adj = to_sparse(adj)
    return adj


