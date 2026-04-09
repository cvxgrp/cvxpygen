
import importlib

import cvxpy as cp
import numpy as np
import scipy.linalg as la
import torch
from cvxpylayers.torch import CvxpyLayer as LayerTorch

from cvxpygen import cpg


def test_regression():
    
    np.random.seed(1)

    # define CVXPY problem
    q, d = 10, 5
    A = np.random.randn(q, d)
    x = cp.Variable(d, name='x')
    b = cp.Parameter(q, name='b')
    obj = cp.sum_squares(A @ x - b)
    constr = [cp.diff(x) >= 0, -1 <= b, b <= 1]
    problem = cp.Problem(cp.Minimize(obj), constr)

    # generate code
    cpg.generate_code(problem, code_dir='explicit_regression', solver='explicit', prefix='ex_regression')
    from explicit_regression.cpg_solver import cpg_solve
    problem.register_solve('cpg_explicit', cpg_solve)
    
    np.random.seed(2)

    b.value = -1 + 2 * np.random.rand(q)
        
    problem.solve(solver='OSQP')
    x_ref = x.value
    obj_ref = obj.value
    
    problem.solve(method='cpg_explicit')
    assert np.allclose(x.value, x_ref)
    assert np.allclose(obj.value, obj_ref)
    
    
def test_power():
    
    np.random.seed(1)
    
    C = 1
    D = 1
    h = 0.05
    Q = 1
    qtar = 0.5
    alpha = 0.1
    beta = 0.1

    g = cp.Variable(name='g')
    s = cp.Variable(name='s')
    b = cp.Variable(name='b')
    qplus = cp.Variable(name='qplus')
    
    L = cp.Parameter(name='L')
    S = cp.Parameter(name='S')
    P = cp.Parameter(name='P')
    q = cp.Parameter(name='q')
    
    obj = P * g * h + alpha * (qplus - qtar)**2 + beta * b**2
    constr = [
            L == s + b + g,
            0 <= s, s <= S, -C <= b, b <= D, g >= 0,
            qplus == q - h * b, 0 <= qplus, qplus <= Q,
            0 <= L, L <= 1,
            0 <= S, S <= 0.5,
            1 <= P, P <= 2,
            0 <= q, q <= Q,
        ]
    
    problem = cp.Problem(cp.Minimize(obj), constr)

    # generate code
    cpg.generate_code(problem, code_dir='explicit_power', solver='explicit', prefix='ex_power')
    from explicit_power.cpg_solver import cpg_solve
    problem.register_solve('cpg_explicit', cpg_solve)

    np.random.seed(2)

    L.value = np.random.rand()
    S.value = 0.5 * np.random.rand()
    P.value = 1 + np.random.rand()
    q.value = Q * np.random.rand()
    
    problem.solve(solver='OSQP')
    g_ref = g.value
    s_ref = s.value
    b_ref = b.value
    qplus_ref = qplus.value
    obj_ref = obj.value
    
    problem.solve(method='cpg_explicit')
    rtol = 1e-4
    assert np.allclose(g.value, g_ref, rtol=rtol)
    assert np.allclose(s.value, s_ref, rtol=rtol)
    assert np.allclose(b.value, b_ref, rtol=rtol)
    assert np.allclose(qplus.value, qplus_ref, rtol=rtol)
    assert np.allclose(obj.value, obj_ref, rtol=rtol)


def test_control():
    
    np.random.seed(1)
    
    n, m = 6, 1
    H = 5
    
    A = 0.1 * np.random.randn(n, n)
    np.fill_diagonal(A, np.random.randn(n))
    evs, _ = np.linalg.eigh(A)
    A /= np.max(np.abs(evs))
    B = np.sqrt(0.001) * np.random.randn(n, m)
    
    Q = np.eye(n)
    R = 0.1 * np.eye(m)
    
    P = la.solve_discrete_are(A, B, Q, R)
    
    X = cp.Variable((n, H+1), name='X')
    U = cp.Variable((m, H), name='U')
    
    xinit = cp.Parameter(n, name='xinit')

    obj = cp.quad_form(X[:, H], P) + cp.sum_squares(X[:, :-1]) + 0.1 * cp.sum_squares(U)
    constr = [
        X[:, 1:] == A @ X[:, :-1] + B @ U,
        -1 <= U, U <= 1,  # TODO: Test use of cp.norm and cp.abs
        X[:, 0] == xinit,
        -1 <= xinit, xinit <= 1
    ]

    problem = cp.Problem(cp.Minimize(obj), constr)
    
    # generate code
    cpg.generate_code(problem, code_dir='explicit_MPC', solver='explicit', prefix='ex_mpc')
    from explicit_MPC.cpg_solver import cpg_solve
    problem.register_solve('cpg_explicit', cpg_solve)

    np.random.seed(2)
    
    xinit.value = -1 + 2 * np.random.rand(n)
    
    problem.solve(solver='OSQP')
    X_ref = X.value
    U_ref = U.value
    obj_ref = obj.value
    
    problem.solve(method='cpg_explicit')
    rtol = 1e-4
    assert np.allclose(X.value, X_ref, rtol=rtol)
    assert np.allclose(U.value, U_ref, rtol=rtol)
    assert np.allclose(obj.value, obj_ref, rtol=rtol)
    
    
def test_control_fp16():
    
    np.random.seed(1)
    
    n, m = 6, 1
    H = 5
    
    A = 0.1 * np.random.randn(n, n)
    np.fill_diagonal(A, np.random.randn(n))
    evs, _ = np.linalg.eigh(A)
    A /= np.max(np.abs(evs))
    B = np.sqrt(0.001) * np.random.randn(n, m)
    
    Q = np.eye(n)
    R = 0.1 * np.eye(m)
    
    P = la.solve_discrete_are(A, B, Q, R)
    
    X = cp.Variable((n, H+1), name='X')
    U = cp.Variable((m, H), name='U')
    
    xinit = cp.Parameter(n, name='xinit')
        
    obj = cp.quad_form(X[:, H], P) + cp.sum_squares(X[:, :-1]) + 0.1 * cp.sum_squares(U)
    constr = [
        X[:, 1:] == A @ X[:, :-1] + B @ U,
        -1 <= U, U <= 1,
        X[:, 0] == xinit,
        -1 <= xinit, xinit <= 1
    ]
        
    problem = cp.Problem(cp.Minimize(obj), constr)
    
    # generate code
    cpg.generate_code(problem, code_dir='explicit_MPC_fp16', solver='explicit', solver_opts={'fp16': True}, prefix='ex_mpc_fp16')
    from explicit_MPC_fp16.cpg_solver import cpg_solve
    problem.register_solve('cpg_explicit', cpg_solve)

    np.random.seed(2)
    
    xinit.value = -1 + 2 * np.random.rand(n)
    
    problem.solve(solver='OSQP')
    X_ref = X.value
    U_ref = U.value
    obj_ref = obj.value
    
    problem.solve(method='cpg_explicit')
    rtol = 1e-3
    assert np.allclose(X.value, X_ref, rtol=rtol)
    assert np.allclose(U.value, U_ref, rtol=rtol)
    assert np.allclose(obj.value, obj_ref, rtol=rtol)


def test_control_reduced():

    np.random.seed(1)

    n, m = 6, 1
    H = 5

    A = 0.1 * np.random.randn(n, n)
    np.fill_diagonal(A, np.random.randn(n))
    evs, _ = np.linalg.eigh(A)
    A /= np.max(np.abs(evs))
    B = np.sqrt(0.001) * np.random.randn(n, m)

    Q = np.eye(n)
    R = 0.1 * np.eye(m)

    P = la.solve_discrete_are(A, B, Q, R)

    X = cp.Variable((n, H+1), name='X')
    U = cp.Variable((m, H), name='U')

    xinit = cp.Parameter(n, name='xinit')

    obj = cp.quad_form(X[:, H], P) + cp.sum_squares(X[:, :-1]) + 0.1 * cp.sum_squares(U)
    constr = [
        X[:, 1:] == A @ X[:, :-1] + B @ U,
        -1 <= U, U <= 1,  # TODO: Test use of cp.norm and cp.abs
        X[:, 0] == xinit,
        -1 <= xinit, xinit <= 1
    ]

    problem = cp.Problem(cp.Minimize(obj), constr)

    solver_opts= {"stored_vars":[U[:,0],X[[1,2],:]]}
    # generate code
    cpg.generate_code(problem, code_dir='explicit_MPC_reduced', solver='explicit', prefix='ex_mpc_red',
                      solver_opts=solver_opts)
    from explicit_MPC_reduced.cpg_solver import cpg_solve
    problem.register_solve('cpg_explicit', cpg_solve)

    np.random.seed(2)

    xinit.value = -1 + 2 * np.random.rand(n)

    problem.solve(solver='OSQP')
    X_ref = X.value
    U_ref = U.value
    obj_ref = obj.value

    problem.solve(method='cpg_explicit')
    rtol = 1e-4

    assert np.allclose(U.value[:,0], U_ref[:,0], rtol=rtol)
    assert np.allclose(U.value[:,1:5], np.zeros(4), rtol=rtol) # Not stored -> zero
    assert np.allclose(X.value[[1,2],:], X_ref[[1,2],:], rtol=rtol)
    assert np.allclose(X.value[[0,3,4,5],:], np.zeros((4,6)), rtol=rtol) # Not stored -> zero


def test_stored_vars():

    np.random.seed(1)
    # define CVXPY problem
    q, d = 5, 8
    A = np.random.randn(q, d)
    X = cp.Variable((2,2,2), name='X')
    xs = cp.Variable(name='xs')
    b = cp.Parameter(q, name='b')
    obj = cp.sum_squares(A @ cp.vec(X,order='F') + np.random.randn(q,1)*xs - b)
    constr = [cp.diff(cp.vec(X,order='F')) >= 0, -1 <= b, b <= 1]
    problem = cp.Problem(cp.Minimize(obj), constr)

    # generate code
    cpg.generate_code(problem, code_dir='ex_store_X', solver='explicit', prefix='ex_store_X', solver_opts = {'stored_vars':[X[0,:,[1]]]})
    from ex_store_X.cpg_solver import cpg_solve
    problem.register_solve('cpg_explicit_X', cpg_solve)

    cpg.generate_code(problem, code_dir='ex_store_xs', solver='explicit', prefix='ex_store_xs', solver_opts = {'stored_vars':[xs]})
    from ex_store_xs.cpg_solver import cpg_solve
    problem.register_solve('cpg_explicit_xs', cpg_solve)

    np.random.seed(2)

    b.value = -1 + 2 * np.random.rand(q)

    problem.solve(solver='OSQP')
    X_ref = X.value
    xs_ref = xs.value
    obj_ref = obj.value

    problem.solve(method='cpg_explicit_X')
    Xv = X.value.reshape((2,2,2),order='F') # Due to cvxpygen 0.6.1 flattening if len(shape) > 2
    assert np.allclose(Xv[0,:,[1]], X_ref[0,:,[1]])
    assert np.allclose(Xv[1,:,0], np.zeros(2))
    assert np.allclose(Xv[1,:,1], np.zeros(2))
    assert np.allclose(Xv[0,:,0], np.zeros(2))
    assert np.allclose(Xv[0,:,0], np.zeros(2))
    assert xs.value is None

    problem.solve(method='cpg_explicit_xs')
    assert X.value is None
    assert np.allclose(xs.value, xs_ref)


def test_dual():

    np.random.seed(1)

    d, p = 2, 3
    X = np.random.randn(p, d)
    l = 0
    u = np.ones(p)

    beta = cp.Variable(d, name='beta')
    v = cp.Variable(name='v')
    y = cp.Parameter(p, name='y')

    obj = cp.sum_squares(X @ beta + v - y)
    constr = [beta >= 0, l <= y, y <= u]

    problem = cp.Problem(cp.Minimize(obj), constr)
    
    cpg.generate_code(problem, solver='explicit', solver_opts={'dual': True}, code_dir='explicit_dual', prefix='ex_dual')
    
    np.random.seed(0)
    
    from explicit_dual.cpg_solver import cpg_solve
    problem.register_solve('gen_explicit', cpg_solve)
    
    y.value = [0.6, 0.8, 0.2]
    
    problem.solve(solver='OSQP')
    v_ref = v.value
    beta_ref = beta.value
    dual_ref = constr[0].dual_value
    obj_ref = obj.value
    
    problem.solve(method='gen_explicit')
    print(v.value, v_ref)
    assert np.allclose(v.value, v_ref)
    assert np.allclose(beta.value, beta_ref)
    assert np.allclose(constr[0].dual_value, dual_ref)
    assert np.allclose(obj.value, obj_ref)


def test_gradient():
    """Gradient of sum(sol_x) w.r.t. parameters via explicit solver matches finite differences."""

    np.random.seed(1)
    q, d = 4, 3
    A = np.random.randn(q, d)
    x = cp.Variable(d, name='x')
    b = cp.Parameter(q, name='b')
    obj = cp.sum_squares(A @ x - b)
    constr = [-1 <= b, b <= 1]
    problem = cp.Problem(cp.Minimize(obj), constr)

    cpg.generate_code(problem, code_dir='explicit_gradient', solver='explicit',
                      gradient=True, prefix='ex_grad')
    from explicit_gradient.cpg_solver import cpg_solve, cpg_gradient
    problem.register_solve('cpg_explicit', cpg_solve)

    np.random.seed(2)
    b.value = -0.5 + np.random.rand(q)
    problem.solve(method='cpg_explicit')

    # Compute explicit gradient: dx = ones → dp = sum over col
    for v in problem.variables():
        v.gradient = np.ones(v.shape)
    cpg_gradient(problem)
    db_cpg = b.gradient.copy()

    # Finite-difference reference
    eps = 1e-4
    b0 = b.value.copy()
    db_fd = np.zeros(q)
    for i in range(q):
        bplus = b0.copy(); bplus[i] += eps
        b.value = bplus
        problem.solve(method='cpg_explicit')
        fplus = sum(float(np.sum(v.value)) for v in problem.variables())

        bminus = b0.copy(); bminus[i] -= eps
        b.value = bminus
        problem.solve(method='cpg_explicit')
        fminus = sum(float(np.sum(v.value)) for v in problem.variables())

        db_fd[i] = (fplus - fminus) / (2 * eps)

    assert np.allclose(db_cpg, db_fd, atol=1e-3)


def test_gradient_reduced():
    """Gradient with partially stored variables: only stored components propagate."""

    np.random.seed(1)
    q, d = 4, 3
    A = np.random.randn(q, d)
    x = cp.Variable(d, name='x')
    b = cp.Parameter(q, name='b')
    obj = cp.sum_squares(A @ x - b)
    constr = [-1 <= b, b <= 1]
    problem = cp.Problem(cp.Minimize(obj), constr)

    # store only x[0] and x[2]
    cpg.generate_code(problem, code_dir='explicit_gradient_reduced', solver='explicit',
                      gradient=True, prefix='ex_grad_red',
                      solver_opts={'stored_vars': [x[[0, 2]]]})
    from explicit_gradient_reduced.cpg_solver import cpg_solve, cpg_gradient
    problem.register_solve('cpg_explicit_red', cpg_solve)

    np.random.seed(2)
    b.value = -0.5 + np.random.rand(q)
    problem.solve(method='cpg_explicit_red')

    # gradient of x[0] + x[2] w.r.t. b  (x[1] not stored → zero in x.value)
    x_grad = np.zeros(d)
    x_grad[[0, 2]] = 1.0
    x.gradient = x_grad
    cpg_gradient(problem)
    db_cpg = b.gradient.copy()

    # finite-difference reference using the same solver
    eps = 1e-4
    b0 = b.value.copy()
    db_fd = np.zeros(q)
    for i in range(q):
        bplus = b0.copy(); bplus[i] += eps
        b.value = bplus
        problem.solve(method='cpg_explicit_red')
        fplus = float(x.value[0]) + float(x.value[2])

        bminus = b0.copy(); bminus[i] -= eps
        b.value = bminus
        problem.solve(method='cpg_explicit_red')
        fminus = float(x.value[0]) + float(x.value[2])

        db_fd[i] = (fplus - fminus) / (2 * eps)

    assert np.allclose(db_cpg, db_fd, atol=1e-3)


def test_gradient_layers():
    """Explicit gradient integrates with cvxpylayers (torch) custom_method interface."""

    np.random.seed(1)
    q, d = 4, 3
    A = np.random.randn(q, d)
    x = cp.Variable(d, name='x')
    b = cp.Parameter(q, name='b')
    obj = cp.sum_squares(A @ x - b)
    constr = [-1 <= b, b <= 1]
    problem = cp.Problem(cp.Minimize(obj), constr)

    cpg.generate_code(problem, code_dir='explicit_gradient_layers', solver='explicit',
                      gradient=True, prefix='ex_grad_layers')
    mod = importlib.import_module('explicit_gradient_layers.cpg_solver')

    np.random.seed(0)
    # b must be within the explicit-solver bounds [-1, 1]
    b.value = -0.7 + 1.4 * np.random.rand(q)

    # Reference layer (cvxpylayers built-in differentiation)
    b_tch = torch.tensor(b.value, requires_grad=True)
    layer_ref = LayerTorch(problem, parameters=[b], variables=[x])
    layer_gen = LayerTorch(problem, parameters=[b], variables=[x],
                           custom_method=(mod.forward, mod.backward))

    def _grad(layer, b_t, solver_args=None):
        b_t = b_t.detach().clone().requires_grad_(True)
        kw = {'solver_args': solver_args} if solver_args else {}
        sol, = layer(b_t, **kw)
        (0.1 * sol.sum()).backward()
        return b_t.grad.detach().numpy()

    grad_ref = _grad(layer_ref, b_tch)
    grad_gen = _grad(layer_gen, b_tch,
                     solver_args={'problem': problem, 'updated_params': ['b']})

    assert np.allclose(grad_ref, grad_gen, atol=1e-3)

    # Second parameter value (also within bounds)
    np.random.seed(3)
    b2_tch = torch.tensor(-0.7 + 1.4 * np.random.rand(q), requires_grad=True)
    grad_ref2 = _grad(layer_ref, b2_tch)
    grad_gen2 = _grad(layer_gen, b2_tch,
                      solver_args={'problem': problem, 'updated_params': ['b']})

    assert np.allclose(grad_ref2, grad_gen2, atol=1e-3)
