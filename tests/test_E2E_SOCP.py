
import pytest
import cvxpy as cp
import numpy as np
import glob
import os
import importlib
import itertools
import pickle
import utils_test
import sys
sys.path.append('../')
from cvxpygen import cpg


def ADP_problem():

    # define dimensions
    n, m = 6, 3

    # define variables
    u = cp.Variable(m, name='u')

    # define parameters
    Rsqrt = cp.Parameter((m, m), name='Rsqrt', diag=True)
    f = cp.Parameter(n, name='f')
    G = cp.Parameter((n, m), name='G')

    # define objective
    objective = cp.Minimize(cp.sum_squares(f + G @ u) + cp.sum_squares(Rsqrt @ u))

    # define constraints
    constraints = [cp.norm(u, 2) <= 1]

    # define problem
    return cp.Problem(objective, constraints)


def assign_data(prob, name, seed):

    np.random.seed(seed)

    if name == 'ADP':

        def dynamics(x):
            # continuous-time dynmaics
            A_cont = np.array([[0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, -x[3], 0, 0],
                               [0, 0, 0, 0, -x[4], 0],
                               [0, 0, 0, 0, 0, -x[5]]])
            mass = 1
            B_cont = np.concatenate((np.zeros((3, 3)),
                                     (1 / mass) * np.diag(x[3:])), axis=0)
            # discrete-time dynamics
            td = 0.1
            return np.eye(6) + td * A_cont, td * B_cont

        state = -2*np.ones(6) + 4*np.random.rand(6)
        Psqrt = np.eye(6)
        A, B = dynamics(state)
        prob.param_dict['Rsqrt'].value = np.sqrt(0.1) * np.eye(3)
        prob.param_dict['f'].value = np.matmul(Psqrt, np.matmul(A, state))
        prob.param_dict['G'].value = np.matmul(Psqrt, B)

    return prob


def get_primal_vec(prob, name):
    if name == 'ADP':
        return prob.var_dict['u'].value


N_RAND = 3

name_solver_style_seed = [['ADP'],
                          ['SCS', 'ECOS'],
                          ['unroll', 'loops'],
                          list(np.arange(N_RAND))]

name_to_prob = {'ADP': ADP_problem()}


@pytest.mark.parametrize('name, solver, style, seed', list(itertools.product(*name_solver_style_seed)))
def test(name, solver, style, seed):

    prob = name_to_prob[name]

    if seed == 0:
        if style == 'unroll':
            cpg.generate_code(prob, code_dir='test_%s_%s_unroll' % (name, solver), solver=solver, unroll=True,
                              prefix='%s_%s_ex' % (name, solver))
            assert len(glob.glob(os.path.join('test_%s_%s_unroll' % (name, solver), 'cpg_module.*'))) > 0
        if style == 'loops':
            cpg.generate_code(prob, code_dir='test_%s_%s_loops' % (name, solver), solver=solver, unroll=False,
                              prefix='%s_%s_im' % (name, solver))
            assert len(glob.glob(os.path.join('test_%s_%s_loops' % (name, solver), 'cpg_module.*'))) > 0

    with open('test_%s_%s_%s/problem.pickle' % (name, solver, style), 'rb') as f:
        prob = pickle.load(f)

    module = importlib.import_module('test_%s_%s_%s.cpg_solver' % (name, solver, style))
    prob.register_solve('CPG', module.cpg_solve)

    prob = assign_data(prob, name, seed)

    val_py, prim_py, dual_py, val_cg, prim_cg, dual_cg, prim_py_norm, dual_py_norm = \
        utils_test.check(prob, solver, name, get_primal_vec)

    if not np.isinf(val_py):
        assert abs((val_cg - val_py) / val_py) < 0.1

    if prim_py_norm > 1e-6:
        assert np.linalg.norm(prim_cg - prim_py, 2) / prim_py_norm < 0.1
    else:
        assert np.linalg.norm(prim_cg, 2) < 1e-3

    if dual_py_norm > 1e-6:
        assert np.linalg.norm(dual_cg - dual_py, 2) / dual_py_norm < 0.1
    else:
        assert np.linalg.norm(dual_cg, 2) < 1e-3
