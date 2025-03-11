import cvxpy as cp
import numpy as np
from cvxpygen import cpg
import time, sys

m, n = 3, 2
x = cp.Variable(n, name='x')
A = cp.Parameter((m, n), name='A', sparsity=((0, 0, 1), (0, 1, 1)))
b = cp.Parameter(m, name='b')
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

np.random.seed(0)
A.value = np.zeros((m, n))
A.value[0, 0] = np.random.randn()
A.value[0, 1] = np.random.randn()
A.value[1, 1] = np.random.randn()
b.value = np.random.randn(m)
problem.solve(solver='QOCO', verbose=True)
print('Primal solution: x = [%.6f, %.6f]' % tuple(x.value))
cpg.generate_code(problem, code_dir='nonneg_LS', solver='QOCO')

# # from nonneg_LS.cpg_solver import cpg_solve
# # problem.register_solve('CPG', cpg_solve)

# solve problem conventionally
t0 = time.time()
val = problem.solve(solver='QOCO')
t1 = time.time()
sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
sys.stdout.write('Objective function value: %.6f\n' % val)

# solve problem with C code via python wrapper
t0 = time.time()
val = problem.solve(method='CPG', updated_params=['A', 'b'], verbose=True)
t1 = time.time()
sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
sys.stdout.write('Objective function value: %.6f\n' % val)