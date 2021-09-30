
import pytest
import cvxpy as cp
import numpy as np
import glob
import os
import sys
sys.path.append('../')
import cvxpygen as cpg


def actuator_problem():

    # define dimensions
    n, m = 8, 3

    # define variables
    u = cp.Variable(n, name='u')
    delta_u = cp.Variable(n, name='delta_u')

    # define parameters
    A = cp.Parameter((m, n), name='A')
    w = cp.Parameter(m, name='w')
    lamb_sm = cp.Parameter(nonneg=True, name='lamb_sm')
    kappa = cp.Parameter(n, nonneg=True, name='kappa')
    u_prev = cp.Parameter(n, name='u_prev')
    u_min = cp.Parameter(n, name='u_min')
    u_max = cp.Parameter(n, name='u_max')

    # define objective
    objective = cp.Minimize(cp.sum_squares(A @ u - w) + lamb_sm * cp.sum_squares(delta_u) + kappa @ cp.abs(u))

    # define constraints
    constraints = [u_min <= u, u <= u_max, delta_u == u - u_prev]

    # define problem
    return cp.Problem(objective, constraints)


def ADP_problem():

    # define dimensions
    n, m = 6, 3

    # define variables
    u = cp.Variable(m, name='u')

    # define parameters
    Rsqrt = cp.Parameter((m, m), name='Rsqrt')
    f = cp.Parameter(n, name='f')
    G = cp.Parameter((n, m), name='G')

    # define objective
    objective = cp.Minimize(cp.sum_squares(f + G @ u) + cp.sum_squares(Rsqrt @ u))

    # define constraints
    constraints = [cp.norm(u, 'inf') <= 1]

    # define problem
    return cp.Problem(objective, constraints)


def MPC_problem():

    # define dimensions
    H, n, m = 10, 6, 3

    # define variables
    U = cp.Variable((m, H), name='U')
    X = cp.Variable((n, H + 1), name='X')

    # define parameters
    Psqrt = cp.Parameter((n, n), name='Psqrt')
    Qsqrt = cp.Parameter((n, n), name='Qsqrt')
    Rsqrt = cp.Parameter((m, m), name='Rsqrt')
    A = cp.Parameter((n, n), name='A')
    B = cp.Parameter((n, m), name='B')
    x_init = cp.Parameter(n, name='x_init')

    # define objective
    objective = cp.Minimize(
        cp.sum_squares(Psqrt @ X[:, H - 1]) + cp.sum_squares(Qsqrt @ X[:, :H]) + cp.sum_squares(Rsqrt @ U))

    # define constraints
    constraints = [X[:, 1:] == A @ X[:, :H] + B @ U,
                   cp.abs(U) <= 1,
                   X[:, 0] == x_init]

    # define problem
    return cp.Problem(objective, constraints)


def portfolio_problem():

    # define dimensions
    n, m = 100, 10

    # define variables
    w = cp.Variable(n, name='w')
    delta_w = cp.Variable(n, name='delta_w')
    f = cp.Variable(m, name='f')

    # define parameters
    a = cp.Parameter(n, name='a')
    F = cp.Parameter((n, m), name='F')
    Sig_f_sqrt = cp.Parameter((m, m), name='Sig_f_sqrt')
    d_sqrt = cp.Parameter(n, name='d_sqrt')
    k_tc = cp.Parameter(n, nonneg=True, name='k_tc')
    k_sh = cp.Parameter(n, nonneg=True, name='k_sh')
    w_prev = cp.Parameter(n, name='w_prev')
    L = cp.Parameter(nonneg=True, name='L')

    # define objective
    objective = cp.Maximize(a @ w
                            - cp.sum_squares(Sig_f_sqrt @ f)
                            - cp.sum_squares(cp.multiply(d_sqrt, w))
                            - k_tc @ cp.abs(delta_w)
                            + k_sh @ cp.minimum(0, w))

    # define constraints
    constraints = [f == F.T @ w,
                   np.ones(n) @ w == 1,
                   cp.norm(w, 1) <= L,
                   delta_w == w - w_prev]

    # define problem
    return cp.Problem(objective, constraints)


@pytest.mark.parametrize('prob, name', [(actuator_problem(), 'actuator'),
                                        (ADP_problem(), 'ADP'),
                                        (MPC_problem(), 'MPC'),
                                        (portfolio_problem(), 'portfolio')])
def test(prob, name):

    cpg.generate_code(prob, code_dir='test_%s_explicit' % name, explicit=True, problem_name='%s_ex' % name)
    assert len(glob.glob(os.path.join('test_%s_explicit' % name, 'cpg_module.*'))) > 0

    cpg.generate_code(prob, code_dir='test_%s_implicit' % name, explicit=False, problem_name='%s_im' % name)
    assert len(glob.glob(os.path.join('test_%s_implicit' % name, 'cpg_module.*'))) > 0
