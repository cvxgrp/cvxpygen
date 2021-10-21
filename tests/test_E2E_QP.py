
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


def assign_data(prob, name, seed):

    np.random.seed(seed)

    if name == 'actuator':

        prob.param_dict['A'].value = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                                               [0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, -1, 1, 1, -1, 1, -1, -1]])
        prob.param_dict['w'].value = np.array([1, 1, 1])
        prob.param_dict['lamb_sm'].value = np.random.rand()
        prob.param_dict['kappa'].value = 0.1 * np.ones(8)
        prob.param_dict['u_prev'].value = np.zeros(8)
        prob.param_dict['u_min'].value = -np.ones(8)
        prob.param_dict['u_max'].value = np.ones(8)

    elif name == 'ADP':

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

    elif name == 'MPC':

        # continuous-time dynmaics
        A_cont = np.concatenate((np.array([[0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]]),
                                           np.zeros((3, 6))), axis=0)
        mass = 1
        B_cont = np.concatenate((np.zeros((3, 3)),
                                 (1 / mass) * np.diag(np.ones(3))), axis=0)

        # discrete-time dynamics
        td = 0.1

        prob.param_dict['A'].value = np.eye(6) + td * A_cont
        prob.param_dict['B'].value = td * B_cont
        prob.param_dict['Psqrt'].value = np.eye(6)
        prob.param_dict['Qsqrt'].value = np.eye(6)
        prob.param_dict['Rsqrt'].value = np.sqrt(0.1) * np.eye(3)
        prob.param_dict['x_init'].value = -2*np.ones(6) + 4*np.random.rand(6)

    elif name == 'portfolio':

        n, m = 100, 10
        gamma = 1
        alpha = np.random.randn(n)
        kappa_tc = 0.01 * np.ones(n)
        kappa_sh = 0.05 * np.ones(n)
        prob.param_dict['a'].value = alpha / gamma
        prob.param_dict['F'].value = np.round(np.random.randn(n, m))
        prob.param_dict['Sig_f_sqrt'].value = np.diag(np.random.rand(m))
        prob.param_dict['d_sqrt'].value = np.random.rand(n)
        prob.param_dict['k_tc'].value = kappa_tc / gamma
        prob.param_dict['k_sh'].value = kappa_sh / gamma
        prob.param_dict['w_prev'].value = np.zeros(n)
        prob.param_dict['L'].value = 1.6

    return prob


N_RAND = 100

name_style_seed = [['actuator', 'ADP', 'MPC', 'portfolio'],
                   ['explicit', 'implicit'],
                   list(np.arange(N_RAND))]

name_to_prob = {'actuator': actuator_problem(),
                'ADP': ADP_problem(),
                'MPC': MPC_problem(),
                'portfolio': portfolio_problem()}


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

    val_py = prob.solve(eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=False, adaptive_rho_interval=int(1e6), warm_start=False)
    val_ex = prob.solve(method='CPG', warm_start=False)

    if not np.isinf(val_py):
        assert abs((val_ex - val_py) / val_py) < 0.1