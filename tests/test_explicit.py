
import pytest
import importlib
import cvxpy as cp
import numpy as np
import scipy.linalg as la

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
    identifier = 'explicit_regression'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit')
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('cpg_explicit', mod.cpg_solve)
    
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
    identifier = 'explicit_power'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit')
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('cpg_explicit', mod.cpg_solve)

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


@pytest.mark.parametrize(
    'constr_type, compute_dual',
    [('bounds', False), ('abs', False), ('norm', False), ('bounds', True), ('norm', True)]
    )
def test_control(constr_type, compute_dual, capsys):
    
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
        X[:, 0] == xinit,
        -1 <= xinit, xinit <= 1
    ]
    if constr_type == 'bounds':
        constr += [-1 <= U, U <= 1]
    elif constr_type == 'abs':
        constr += [cp.abs(U) <= 1]
    elif constr_type == 'norm':
        constr += [cp.norm(U, 'inf', axis=0) <= 1]

    problem = cp.Problem(cp.Minimize(obj), constr)
    
    # generate code
    identifier = f'explicit_control_{constr_type}_{"dual" if compute_dual else "nodual"}'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit',
                      solver_opts={'dual': compute_dual})
    captured = capsys.readouterr()
    assert '10 linear inequality constraints' in captured.out
    
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('cpg_explicit', mod.cpg_solve)

    np.random.seed(2)
    
    xinit.value = -1 + 2 * np.random.rand(n)
    
    problem.solve(solver='OSQP')
    X_ref = X.value
    U_ref = U.value
    obj_ref = obj.value
    dual_ref = constr[-1].dual_value.copy()
    
    problem.solve(method='cpg_explicit')
    rtol = 1e-4
    assert np.allclose(X.value, X_ref, rtol=rtol)
    assert np.allclose(U.value, U_ref, rtol=rtol)
    assert np.allclose(obj.value, obj_ref, rtol=rtol)
    
    if compute_dual:
        assert np.allclose(constr[-1].dual_value, dual_ref, rtol=rtol)
    
    
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
    identifier = 'explicit_control_fp16'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit', solver_opts={'fp16': True})
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('cpg_explicit', mod.cpg_solve)

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
        -1 <= U, U <= 1,
        X[:, 0] == xinit,
        -1 <= xinit, xinit <= 1
    ]

    problem = cp.Problem(cp.Minimize(obj), constr)

    # generate code
    identifier = 'explicit_control_reduced'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit',
                      solver_opts={"stored_vars":[U[:,0],X[[1,2],:]]})
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('cpg_explicit', mod.cpg_solve)

    np.random.seed(2)

    xinit.value = -1 + 2 * np.random.rand(n)

    problem.solve(solver='OSQP')
    X_ref = X.value
    U_ref = U.value

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
    identifier = 'explicit_stored_vars_X'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit',
                      solver_opts = {'stored_vars':[X[0,:,[1]]]})
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('cpg_explicit_X', mod.cpg_solve)

    identifier = 'explicit_stored_vars_xs'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit',
                      solver_opts = {'stored_vars':[xs]})
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('cpg_explicit_xs', mod.cpg_solve)

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
    
    # generate code
    identifier = 'explicit_dual'
    cpg.generate_code(problem, code_dir=identifier, prefix=identifier, solver='explicit', solver_opts={'dual': True})    
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    problem.register_solve('gen_explicit', mod.cpg_solve)
    
    y.value = [0.6, 0.8, 0.2]
    
    problem.solve(solver='OSQP')
    v_ref = v.value
    beta_ref = beta.value
    dual_ref = constr[0].dual_value
    obj_ref = obj.value
    
    problem.solve(method='gen_explicit')
    assert np.allclose(v.value, v_ref)
    assert np.allclose(beta.value, beta_ref)
    assert np.allclose(constr[0].dual_value, dual_ref)
    assert np.allclose(obj.value, obj_ref)
