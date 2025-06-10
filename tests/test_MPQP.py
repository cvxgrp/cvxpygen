
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
    
    cpg.generate_code(problem, solver='explicit_primal_dual', code_dir='explicit_primal_dual', prefix='ex_dual')
    
    np.random.seed(0)
    
    from explicit_primal_dual.cpg_solver import cpg_solve
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
