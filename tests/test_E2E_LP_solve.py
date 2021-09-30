
import pytest
import importlib
import pickle
import itertools
import numpy as np
import sys
sys.path.append('../')


N_RAND = 10

name_style_seed = [['network', 'resource'],
                   ['explicit', 'implicit'],
                   list(np.arange(N_RAND))]


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


@pytest.mark.parametrize('name, style, seed', list(itertools.product(*name_style_seed)))
def test(name, style, seed):

    with open('test_%s_%s/problem.pickle' % (name, style), 'rb') as f:
        prob = pickle.load(f)

    module = importlib.import_module('test_%s_%s.cpg_solver' % (name, style))
    prob.register_solve('CPG', module.cpg_solve)

    prob = assign_data(prob, name, seed)

    val_py = prob.solve(solver='OSQP', eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=False, adaptive_rho_interval=int(1e6), warm_start=False)
    val_ex = prob.solve(method='CPG', warm_start=False)

    if not np.isinf(val_py):
        assert abs((val_ex - val_py) / val_py) < 0.1
