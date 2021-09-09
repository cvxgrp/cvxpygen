
import cvxpy as cp
import cvxpygen as cpg
import numpy as np

# define dimensions, variables, parameters
# IMPORTANT: specify variable and parameter names, as they are used in the generated C code
m, n = 3, 2
W = cp.Variable((n, n), name='W')
x = cp.Variable(n, name='x')
y = cp.Variable(name='y')
A = cp.Parameter((m, n), name='A')
b = cp.Parameter(m, name='b')
c = cp.Parameter(nonneg=True, name='c')

# define objective & constraints
objective = cp.Minimize(cp.sum_squares(A @ x - b) + c * cp.sum_squares(x) + cp.sum_squares(y) + cp.sum_squares(W))
constraints = [0 <= x, x <= 1]

# define problem
prob = cp.Problem(objective, constraints)

# generate code
cpg.generate_code(prob, code_dir='CPG_code')
