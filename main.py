
import cvxpy as cp
import gen as cpg  # 'gen' will be 'cvxpygen'

# define dimensions, variables, parameters
m, n = 3, 2
x = cp.Variable((n, 1))
F = cp.Parameter((m, n), name='F')
g = cp.Parameter((m, 1), name='g')
e = cp.Parameter((n, 1), name='e')
gamma = cp.Parameter(nonneg=True, name='gamma')

# define objective & constraints
objective = cp.sum_squares(F @ x - g) + gamma * cp.sum_squares(x)
constraints = [cp.abs(x) <= e]

# define problem
prob = cp.Problem(cp.Minimize(objective), constraints)

# generate code
cpg.generate_code(prob, code_dir='cpg_code', compile=True)





# DEV: plausibility check
'''
import numpy as np
np.random.seed(26)
gamma.value = np.random.rand()
F.value = np.random.rand(m, n)
g.value = np.random.rand(m, 1)
e.value = np.random.rand(n, 1)
prob.solve()
'''
