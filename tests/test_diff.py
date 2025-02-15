
import cvxpy as cp
import numpy as np
import torch
import jax
import jax.numpy as jnp
from cvxpylayers.torch import CvxpyLayer as LayerTorch
from cvxpylayers.jax import CvxpyLayer as LayerJax
from cvxpygen import cpg


def test_torch():

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
    cpg.generate_code(prob, code_dir="code_torch", solver='OSQP', prefix='torch', gradient=True, wrapper=True)
    from code_torch.cpg_solver import forward, backward
    
    # torch function
    A_tch = torch.tensor(A.value, requires_grad=True)
    b_tch = torch.tensor(b.value, requires_grad=True)
    layer_torch = LayerTorch(prob, parameters=[A, b], variables=[x])
    layer_torch_gen = LayerTorch(prob, parameters=[A, b], variables=[x], custom_method=(forward, backward))
    
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
    cpg.generate_code(prob, code_dir="code_jax", solver='OSQP', prefix='jax', gradient=True, wrapper=True)
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
