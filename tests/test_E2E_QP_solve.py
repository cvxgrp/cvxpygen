
import pytest
import importlib
import pickle
import itertools
import numpy as np
import sys
sys.path.append('../')


N_RAND = 100

name_style_seed = [['actuator', 'ADP', 'MPC', 'portfolio'],
                   ['explicit', 'implicit'],
                   list(np.arange(N_RAND))]


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


@pytest.mark.parametrize('name, style, seed', list(itertools.product(*name_style_seed)))
def test(name, style, seed):

    with open('test_%s_%s/problem.pickle' % (name, style), 'rb') as f:
        prob = pickle.load(f)

    module = importlib.import_module('test_%s_%s.cpg_solver' % (name, style))
    prob.register_solve('CPG', module.cpg_solve)

    prob = assign_data(prob, name, seed)

    val_py = prob.solve(eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=False, adaptive_rho_interval=int(1e6), warm_start=False)
    val_ex = prob.solve(method='CPG', warm_start=False)

    if not np.isinf(val_py):
        assert abs((val_ex - val_py) / val_py) < 0.1
