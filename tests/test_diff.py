
import cvxpy as cp
import numpy as np
import torch
import importlib
import pytest
import jax
import jax.numpy as jnp
from cvxpylayers.torch import CvxpyLayer as LayerTorch
from cvxpylayers.jax import CvxpyLayer as LayerJax
from cvxpygen import cpg


@pytest.mark.parametrize("n, solver", [(1, 'OSQP'), (3, 'OSQP'), (3, 'explicit')])
def test_gradient(n, solver):
    
    np.random.seed(0)
    
    # parametrized problem
    x = cp.Variable(n, name='x')
    chol = np.random.randn(n, n)
    Q = chol @ chol.T / 10 + np.eye(n)
    b = cp.Parameter(n, name='b')
    constr = [x >= 0]
    if solver == 'explicit':
        constr += [-1 <= b, b <= 1]
    obj = cp.Minimize(b @ x + cp.quad_form(x, Q))
    prob = cp.Problem(obj, constr)
    b.value = -1 + 2 * np.random.rand(n)
    
    # generate code
    identifier = f'gradient_{n}_{solver}'
    cpg.generate_code(prob, code_dir=identifier, solver=solver, prefix=identifier, gradient=True)
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    prob.register_solve('cpg', mod.cpg_solve)

    # assign parameter value
    b.value = -0.5 + np.random.rand(n)

    # compute gradients
    prob.solve(method='cpg')
    x.gradient = np.ones(x.shape)
    mod.cpg_gradient(prob)
    g_cpg = b.gradient
    
    # cvxpy reference
    prob.solve(requires_grad=True)
    prob.backward()
    g_cvxpy = b.gradient

    assert np.allclose(g_cpg, g_cvxpy)
    
    
@pytest.mark.parametrize("n, solver", [(5, 'OSQP'), (5, 'explicit')])
def test_torch_sim(n, solver):
    
    np.random.seed(0)
    
    # parametrized nonneg LS problem
    x = cp.Variable(n, name='x')
    xhat = cp.Parameter(n, name='xhat')
    b = cp.Parameter(n, name='b')
    constr = [0.0 <= x, x <= 1.0, x >= xhat]
    if solver == 'explicit':
        constr += [0.0 <= xhat, xhat <= 1.0, 0.0 <= b, b <= 1.0]
    obj = cp.Minimize(cp.sum_squares(x - b))
    prob = cp.Problem(obj, constr)
    
    xhat.value = np.random.rand(n)
    b.value = np.random.rand(n)
    
    # generate code
    identifier = f'torch_sim_{n}_{solver}'
    cpg.generate_code(prob, code_dir=identifier, solver=solver, prefix=identifier, gradient=True)
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    prob.register_solve('cpg', mod.cpg_solve)
    
    # torch function
    layer = LayerTorch(prob, parameters=[xhat, b], variables=[x])
    layer_gen = LayerTorch(prob, parameters=[xhat, b], variables=[x], custom_method=(mod.forward, mod.backward))
    
    def sim(lyr, xhat0, solver_args={}):
        
        np.random.seed(0)
        b0 = torch.tensor(np.random.rand(n))
        xhat1, = lyr(xhat0, b0, solver_args=solver_args)
                
        b1 = torch.tensor(np.random.rand(n))
        xhat2, = lyr(xhat1, b1, solver_args=solver_args)
                
        b2 = torch.tensor(np.random.rand(n))
        xhat3, = lyr(xhat2, b2, solver_args=solver_args)
                
        return xhat3.sum()
    
    xhat0_tch = torch.tensor(xhat.value, requires_grad=True)
    
    res = sim(layer, xhat0_tch)
    res.backward()
    grad = xhat0_tch.grad.detach().numpy()
    
    res_gen = sim(layer_gen, xhat0_tch, solver_args={'problem': prob, 'updated_params': ['xhat', 'b']})
    res_gen.backward()
    grad_gen = xhat0_tch.grad.detach().numpy()
    
    assert np.allclose(grad, grad_gen)


@pytest.mark.parametrize("m, n, solver", [(10, 5, 'QOCOGEN'), (10, 5, 'CLARABEL'), (10, 5, 'SCS')])
def test_torch_two_stage(m, n, solver):

    # parametrized nonneg LS problem
    x = cp.Variable(n, nonneg=True, name='x')
    A = cp.Parameter((m, n), name='A')
    b = cp.Parameter(m, name='b')
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
    
    np.random.seed(0)
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    
    # generate code
    identifier = f'torch_two_stage_{m}_{n}_{solver}'
    cpg.generate_code(prob, code_dir=identifier, solver=solver, prefix=identifier, gradient=True)
    mod = importlib.import_module(f'{identifier}.cpg_solver')
    
    # torch function
    A_tch = torch.tensor(A.value, requires_grad=True)
    b_tch = torch.tensor(b.value, requires_grad=True)
    layer_torch = LayerTorch(prob, parameters=[A, b], variables=[x])
    layer_torch_gen = LayerTorch(prob, parameters=[A, b], variables=[x], custom_method=(mod.forward, mod.backward))
    
    sol_torch, = layer_torch(A_tch, b_tch)
    sum_torch = 0.1 * sol_torch.sum()
    sum_torch.backward()
    grad_A_torch = A_tch.grad.detach().numpy()
    grad_b_torch = b_tch.grad.detach().numpy()
    
    solver_args={'problem': prob, 'updated_params': ['A', 'b']}
    if solver == 'SCS':
        solver_args['verbose'] = False
    sol_torch_gen, = layer_torch_gen(A_tch, b_tch, solver_args=solver_args)
    sum_torch_gen = 0.1 * sol_torch_gen.sum()
    sum_torch_gen.backward()
    grad_A_torch_gen = A_tch.grad.detach().numpy()
    grad_b_torch_gen = b_tch.grad.detach().numpy()
    
    assert np.allclose(grad_A_torch, grad_A_torch_gen)
    assert np.allclose(grad_b_torch, grad_b_torch_gen)
    
    # change parameter values
    A_tch = torch.tensor(np.random.rand(m, n), requires_grad=True)
    b_tch = torch.tensor(np.random.rand(m), requires_grad=True)
    
    sol_torch, = layer_torch(A_tch, b_tch)
    sum_torch = 0.1 * sol_torch.sum()
    sum_torch.backward()
    grad_A_torch = A_tch.grad.detach().numpy()
    grad_b_torch = b_tch.grad.detach().numpy()
    
    sol_torch_gen, = layer_torch_gen(A_tch, b_tch, solver_args={'problem': prob, 'updated_params': ['A', 'b']})
    sum_torch_gen = 0.1 * sol_torch_gen.sum()
    sum_torch_gen.backward()
    grad_A_torch_gen = A_tch.grad.detach().numpy()
    grad_b_torch_gen = b_tch.grad.detach().numpy()
    
    assert np.allclose(grad_A_torch, grad_A_torch_gen)
    assert np.allclose(grad_b_torch, grad_b_torch_gen)
    

def test_jax():
    
    # parametrized nonneg LS problem
    m, n = 10, 5
    x = cp.Variable(n, nonneg=True, name='x')
    A = cp.Parameter((m, n), name='A')
    b = cp.Parameter(m, name='b')
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
    
    np.random.seed(0)
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    
    # generate code
    cpg.generate_code(prob, code_dir="code_jax", solver='OSQP', prefix='jax', gradient=True)
    from code_jax.cpg_solver import forward, backward
    
    # jax function
    A_jax = jax.device_put(jnp.array(A.value))
    b_jax = jax.device_put(jnp.array(b.value))
    layer_jax = LayerJax(prob, parameters=[A, b], variables=[x])
    layer_jax_gen = LayerJax(prob, parameters=[A, b], variables=[x], custom_method=(forward, backward))
    
    def func(A_jax, b_jax):
        sol = layer_jax(A_jax, b_jax)[0]
        return 0.1 * jnp.sum(sol)
    
    def func_gen(A_jax, b_jax):
        sol = layer_jax_gen(A_jax, b_jax, solver_args={'problem': prob, 'updated_params': ['A', 'b']})[0]
        return 0.1 * jnp.sum(sol)
    
    grad_A_jax, grad_b_jax = jax.grad(func, argnums=(0, 1))(A_jax, b_jax)
    grad_A_jax_gen, grad_b_jax_gen = jax.grad(func_gen, argnums=(0, 1))(A_jax, b_jax)
        
    assert np.allclose(grad_A_jax, grad_A_jax_gen, atol=1e-4)
    assert np.allclose(grad_b_jax, grad_b_jax_gen, atol=1e-4)
    
    # change parameter values
    A_jax = jax.device_put(jnp.array(np.random.rand(m, n)))
    b_jax = jax.device_put(jnp.array(np.random.rand(m)))
    
    grad_A_jax, grad_b_jax = jax.grad(func, argnums=(0, 1))(A_jax, b_jax)
    grad_A_jax_gen, grad_b_jax_gen = jax.grad(func_gen, argnums=(0, 1))(A_jax, b_jax)
    
    assert np.allclose(grad_A_jax, grad_A_jax_gen, atol=1e-4)
    assert np.allclose(grad_b_jax, grad_b_jax_gen, atol=1e-4)


def test_explicit_reduced():
    """Gradient with partially stored variables: only stored components propagate."""

    np.random.seed(1)
    m, n = 4, 3
    A = np.random.randn(m, n)
    x = cp.Variable(n, name='x')
    b = cp.Parameter(m, name='b')
    obj = cp.sum_squares(A @ x - b)
    constr = [-1 <= b, b <= 1]
    prob = cp.Problem(cp.Minimize(obj), constr)

    # store only x[0] and x[2]
    identifier = 'explicit_gradient_reduced'
    cpg.generate_code(prob, code_dir=identifier, solver='explicit', gradient=True,
                      prefix=identifier, solver_opts={'stored_vars': [x[[0, 2]]]})
    from explicit_gradient_reduced.cpg_solver import cpg_solve, cpg_gradient
    prob.register_solve('cpg_explicit_red', cpg_solve)

    np.random.seed(2)
    b.value = -0.5 + np.random.rand(m)
    prob.solve(method='cpg_explicit_red')

    # gradient of x[0] + x[2] w.r.t. b  (x[1] not stored → zero in x.value)
    x_grad = np.zeros(n)
    x_grad[[0, 2]] = 1.0
    x.gradient = x_grad
    cpg_gradient(prob)
    db_cpg = b.gradient.copy()

    # Finite-difference reference
    eps = 1e-4
    b0 = b.value.copy()
    db_fd = np.zeros(m)
    for i in range(m):
        bplus = b0.copy()
        bplus[i] += eps
        b.value = bplus
        prob.solve(method='cpg_explicit_red')
        fplus = np.sum(x.value)

        bminus = b0.copy()
        bminus[i] -= eps
        b.value = bminus
        prob.solve(method='cpg_explicit_red')
        fminus = np.sum(x.value)

        db_fd[i] = (fplus - fminus) / (2 * eps)

    assert np.allclose(db_cpg, db_fd, atol=1e-3)
