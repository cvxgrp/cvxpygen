
import cvxpy as cp
import cvxpygen as cpg
import numpy as np
import time

'''
1. Generate Code
'''

# define CVXPY problem
m, n = 3, 2
x = cp.Variable(n, name='x')
A = cp.Parameter((m, n), name='A', sparsity=[(0, 0), (0, 1), (1, 1)])
b = cp.Parameter(m, name='b')
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

# assign parameter values and test-solve
np.random.seed(0)
A.value = np.zeros((m, n))
A.value[0, 0] = np.random.randn()
A.value[0, 1] = np.random.randn()
A.value[1, 1] = np.random.randn()
b.value = np.random.randn(m)
problem.solve(solver='SCS')

# generate code
cpg.generate_code(problem, code_dir='nonneg_LS', solver='SCS')

'''
2. Solve & Compare
'''

# import extension module and register custom CVXPY solve method
from nonneg_LS.cpg_solver import cpg_solve
problem.register_solve('cpg', cpg_solve)

# solve problem conventionally
t0 = time.time()
val = problem.solve(solver='SCS')
t1 = time.time()
print('\nStandard method\nSolve time:', np.round(1000*(t1-t0), 3), 'ms')
print('Primal solution: x = ', x.value)
print('Dual solution: d0 = ', problem.constraints[0].dual_value)
print('Objective function value:', val)

# solve problem with C code via python wrapper
t0 = time.time()
val = problem.solve(method='cpg', updated_params=['A', 'b'], verbose=False)
t1 = time.time()
print('\nCodegen method\nSolve time:', np.round(1000*(t1-t0), 3), 'ms')
print('Primal solution: x = ', x.value)
print('Dual solution: d0 = ', problem.constraints[0].dual_value)
print('Objective function value:', val)
