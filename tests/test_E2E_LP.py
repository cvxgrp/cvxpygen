
import pytest
import cvxpy as cp
import numpy as np
import glob
import os
import importlib
import itertools
import pickle
import sys
sys.path.append('../')
import cvxpygen as cpg


def network_problem():

    # define dimensions
    n, m = 100, 10

    # define variable
    f = cp.Variable(n, name='f')

    # define parameters
    R = cp.Parameter((m, n), name='R')
    c = cp.Parameter(m, nonneg=True, name='c')
    w = cp.Parameter(n, nonneg=True, name='w')
    f_min = cp.Parameter(n, nonneg=True, name='f_min')
    f_max = cp.Parameter(n, nonneg=True, name='f_max')

    # define objective
    objective = cp.Maximize(w @ f)

    # define constraints
    constraints = [R @ f <= c, f_min <= f, f <= f_max]

    # define problem
    return cp.Problem(objective, constraints)


def resource_problem():

    # define dimensions
    n, m = 30, 10

    # define variable
    X = cp.Variable((n, m), name='X')

    # define parameters
    W = cp.Parameter((n, m), name='W')
    S = cp.Parameter((n, n), diag=True, name='S')
    X_min = cp.Parameter((n, m), name='X_min')
    X_max = cp.Parameter((n, m), name='X_max')
    r = cp.Parameter(m, name='r')

    # define objective
    objective = cp.Maximize(cp.trace(cp.minimum(X @ W.T, S)))

    # define constraints
    constraints = [X_min <= X, X <= X_max,
                   X.T @ np.ones(n) <= r]

    # define problem
    return cp.Problem(objective, constraints)


def assign_data(prob, name, seed):

    np.random.seed(seed)

    if name == 'network':

        n, m = 100, 10
        prob.param_dict['R'].value = np.round(np.random.rand(m, n))
        prob.param_dict['c'].value = n * (0.1 + 0.1 * np.random.rand(m))
        prob.param_dict['w'].value = np.random.rand(n)
        prob.param_dict['f_min'].value = np.zeros(n)
        prob.param_dict['f_max'].value = np.ones(n)

    elif name == 'resource':

        n, m = 30, 10
        prob.param_dict['S'].value = 100 * np.eye(n)
        prob.param_dict['W'].value = np.ones((n, m)) + 0.1 * np.random.rand(n, m)
        prob.param_dict['X_min'].value = np.random.rand(n, m)
        prob.param_dict['X_max'].value = 10 + np.random.rand(n, m)
        prob.param_dict['r'].value = np.matmul(prob.param_dict['X_min'].value.T, np.ones(n)) + 10 * np.random.rand(m)

    return prob


N_RAND = 10

name_style_seed = [['network', 'resource'],
                   ['explicit', 'implicit'],
                   list(np.arange(N_RAND))]


name_to_prob = {'network': network_problem(), 'resource': resource_problem()}


@pytest.mark.parametrize('name, style, seed', list(itertools.product(*name_style_seed)))
def test(name, style, seed):

    prob = name_to_prob[name]

    if seed == 0:
        if style == 'explicit':
            cpg.generate_code(prob, code_dir='test_%s_explicit' % name, explicit=True, problem_name='%s_ex' % name)
            assert len(glob.glob(os.path.join('test_%s_explicit' % name, 'cpg_module.*'))) > 0
        if style == 'implicit':
            cpg.generate_code(prob, code_dir='test_%s_implicit' % name, explicit=False, problem_name='%s_im' % name)
            assert len(glob.glob(os.path.join('test_%s_implicit' % name, 'cpg_module.*'))) > 0

    with open('test_%s_%s/problem.pickle' % (name, style), 'rb') as f:
        prob = pickle.load(f)

    module = importlib.import_module('test_%s_%s.cpg_solver' % (name, style))
    prob.register_solve('CPG', module.cpg_solve)

    prob = assign_data(prob, name, seed)

    val_py = prob.solve(solver='OSQP', eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=False, adaptive_rho_interval=int(1e6), warm_start=False)
    val_ex = prob.solve(method='CPG', warm_start=False)

    if not np.isinf(val_py):
        assert abs((val_ex - val_py) / val_py) < 0.1
