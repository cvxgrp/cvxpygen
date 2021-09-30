
import pytest
import cvxpy as cp
import numpy as np
import glob
import os
import sys
sys.path.append('../')
import cvxpygen as cpg


def charging_problem():

    # define dimension
    T = 1440

    # define variables
    u = cp.Variable(T, name='u')
    q = cp.Variable(T + 1, name='q')

    # define parameters
    p = cp.Parameter(T, nonneg=True, name='p')
    s = cp.Parameter(T, nonneg=True, name='s')
    D = cp.Parameter(nonneg=True, name='D')
    C = cp.Parameter(nonneg=True, name='C')
    Q = cp.Parameter(nonneg=True, name='Q')

    # define objective
    objective = cp.Minimize(p @ u + s @ cp.abs(u))

    # define constraints
    constraints = [q[1:] == q[:-1] + u,
                   -D <= u, u <= C,
                   0 <= q, q <= Q,
                   q[0] == 0, q[-1] == Q]

    # define problem
    return cp.Problem(objective, constraints)


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


@pytest.mark.parametrize('prob, name', [(charging_problem(), 'charging'),
                                        (network_problem(), 'network'),
                                        (resource_problem(), 'resource')])
def test(prob, name):

    cpg.generate_code(prob, code_dir='test_%s_explicit' % name, explicit=True, problem_name='%s_ex' % name)
    assert len(glob.glob(os.path.join('test_%s_explicit' % name, 'cpg_module.*'))) > 0

    cpg.generate_code(prob, code_dir='test_%s_implicit' % name, explicit=False, problem_name='%s_im' % name)
    assert len(glob.glob(os.path.join('test_%s_implicit' % name, 'cpg_module.*'))) > 0
