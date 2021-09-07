
import cvxpy as cp
import gen as cpg  # 'gen' will be 'cvxpygen'

# define dimensions, variables, parameters
m, n = 3, 2
x = cp.Variable((n, 1), name='x')
y = cp.Variable((n, 1), name='y')
F = cp.Parameter((m, n), name='F')
g = cp.Parameter((m, 1), name='g')
e = cp.Parameter((n, 1), name='e')
delta = cp.Parameter(nonneg=True, name='delta')

# define objective & constraints
objective = cp.sum_squares(F @ (x-2*y) - g) + delta * cp.sum_squares(x)
constraints = [cp.abs(x) <= e, cp.abs(y) <= 2*e]

# define problem
prob = cp.Problem(cp.Minimize(objective), constraints)

# generate code
cpg.generate_code(prob, code_dir='CPG_code')
