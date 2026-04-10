
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


@pytest.mark.parametrize("m, n", [(10, 5), (1, 1)])
def test_torch(m, n):

    # parametrized nonneg LS problem
    x = cp.Variable(n, nonneg=True, name='x')
    A = cp.Parameter((m, n), name='A')
    b = cp.Parameter(m, name='b')
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
    
    np.random.seed(0)
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    
    # generate code
    cpg.generate_code(prob, code_dir=f'code_torch_{m}_{n}', solver='OSQP', prefix=f'torch_{m}_{n}', gradient=True)
    mod = importlib.import_module(f'code_torch_{m}_{n}.cpg_solver')
    
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
    
    sol_torch_gen, = layer_torch_gen(A_tch, b_tch, solver_args={'problem': prob, 'updated_params': ['A', 'b']})
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


def test_explicit():
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


def test_explicit_reduced():
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


def test_explicit_layers():
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
