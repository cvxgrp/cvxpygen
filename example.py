
import cvxpy as cp
import cvxpygen as cpg
import numpy as np
import pickle
import time

'''
1. Generate Code
'''

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

# assign parameter values and solve
np.random.seed(0)
A.value = np.random.randn(m, n)
b.value = np.random.randn(m)
c.value = np.random.rand()
val = prob.solve()
print('Solution: x = ', x.value)
print('Objective function value:', val)

# generate code
cpg.generate_code(prob, code_dir='CPG_code', explicit=True)

'''
2. Solve & Compare
'''

# import generated extension module
from CPG_code.cpg_solver import cpg_solve

# load the serialized problem formulation
with open('CPG_code/problem.pickle', 'rb') as f:
    prob = pickle.load(f)

# assign parameter values
np.random.seed(0)
prob.param_dict['A'].value = np.random.randn(3, 2)
prob.param_dict['b'].value = np.random.randn(3,)
prob.param_dict['c'].value = np.random.rand()

# solve problem conventionally
t0 = time.time()
# CVXPY chooses eps_abs=eps_rel=1e-5, max_iter=10000, polish=True by default,
# however, we choose the OSQP default values here, as they are used for code generation as well
val = prob.solve(eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=False)
t1 = time.time()
print('\nPython solve time:', 1000*(t1-t0), 'ms')
print('Python solution: x = ', prob.var_dict['x'].value)
print('Python objective function value:', val)

# solve problem with C code via python wrapper
prob.register_solve('CPG', cpg_solve)
t0 = time.time()
# the argument 'updated_params' specifies which user-defined parameter values are new
# if the argument is omitted, all values are assumed to be new
# if only a subset of the user-defined parameters have new values, use this argument to speed up the solver
val = prob.solve(method='CPG', eps_abs=1e-3, updated_params=['A', 'b', 'c'])
t1 = time.time()
print('\nC solve time:', 1000*(t1-t0), 'ms')
print('C solution: x = ', prob.var_dict['x'].value)
print('C objective function value:', val)
