import time
import numpy as np
import scipy as sp
import cvxpy as cp
from cvxpygen import cpg


# define problem
m, n = 3, 2
x = cp.Variable(n, name='x')  # name will be used in generated code
A = cp.Parameter((m, n), name='A', sparsity=((0, 0, 1), (0, 1, 1)))
b = cp.Parameter(m, name='b')
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

# generate code (automatically registers the custom 'CPG' method)
cpg.generate_code(problem, code_dir='nonneg_ls')

# assign parameter values
np.random.seed(1)
A.value_sparse = sp.sparse.coo_array((np.random.randn(3), A.sparse_idx), shape=(m, n))
b.value = np.random.randn(m)

# solve with CVXPY
problem.solve(solver=cp.OSQP)
t = time.time()  # time second solve for fair comparison
v_ref = problem.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, polish=False)  # match code gen settings
t_ref = time.time() - t
x_ref = x.value.copy()
d_ref = problem.constraints[0].dual_value.copy()

# solve with CVXPYgen
t = time.time()
v_gen = problem.solve(method='CPG')  # if only, say, b is updated, set updated_params=['b']
t_gen = time.time() - t
x_gen = x.value.copy()
d_gen = problem.constraints[0].dual_value.copy()

# print comparison
print(f'\n\t\tCVXPY\t\tCVXPYgen')
print(f'Solve time\t{1e3 * t_ref:.2f} ms\t\t{1e3 * t_gen:.2f} ms')
print(f'Objective\t{v_ref:.4f}\t\t{v_gen:.4f}')
print(f'Solution\t{np.round(x_ref, 4)}\t{np.round(x_gen, 4)}')
print(f'Dual solution\t{np.round(d_ref, 4)}\t{np.round(d_gen, 4)}')
