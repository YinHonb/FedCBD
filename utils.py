import numpy as np
import cvxpy as cp
import copy
import torch


def compute_cos_matrix(args, model, initial_global_parameters, dw):
    model_temp = {k: model[k] for k in range(len(model))}
    model_similarity_matrix = torch.zeros((len(model_temp),len(model_temp)))
    index_clientid = list(model_temp.keys())
    for i in range(len(model_temp)):
        model_i = model_temp[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] =  model_i[key] - initial_global_parameters[key]
    for i in range(len(model_temp)):
        for j in range(i, len(model_temp)):
            diff = torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
            model_similarity_matrix[i, j] = diff
            model_similarity_matrix[j, i] = diff

    return model_similarity_matrix

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def optimizing_weight_matrix(cos_matrix, p_vector, alpha):
    w_matrix = []
    n = cos_matrix.shape[0]
    P = cp.atoms.affine.wraps.psd_wrap(np.identity(n))
    ones_matrix = np.identity(n)
    zero_vector = np.zeros(n)
    ones_vector = np.ones((1, n))
    ones = np.ones(1)
    for i in range(n):
        cos_vector = copy.deepcopy(cos_matrix[i])

        # =max
        temp = copy.deepcopy(cos_vector)
        before_i = temp[:i]
        after_i = temp[i + 1:]
        temp = torch.cat((before_i, after_i))
        cos_vector[i] = max(temp)

        cv = compute_cv(cos_vector, i)
        if cv <= 0.1:
            cv = 0

        x = cp.Variable(n)
        temp = cos_vector * alpha * cv + alpha * p_vector
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - temp.T @ x),
                          [ones_matrix @ x >= zero_vector,
                           ones_vector @ x == ones]
                          )
        prob.solve()
        w_matrix.append(x.value)
    return w_matrix


def compute_cv(cos_vector, node):
    temp = copy.deepcopy(cos_vector)
    temp[node] = np.nan
    return np.nanstd(temp) / (np.nanmean(temp) + 0.1)

