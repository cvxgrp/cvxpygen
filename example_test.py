
from CPG_code.cpg_solver import cpg_solve
import numpy as np
import pickle
import time

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
# CVXPY chooses eps_abs=eps_rel=1e-5, max_iter=4000, polish=True by default,
# however, we choose the OSQP default values here, as they are used for code generation as well
val = prob.solve(eps_abs=1e-3, eps_rel=1e-3, max_iter=10000, polish=False)
t1 = time.time()
print('\nPython solve time:', 1000*(t1-t0), 'ms')
print('Python solution: x = ', prob.var_dict['x'].value)
print('Python objective function value:', val)

# solve problem with C code via python wrapper
prob.register_solve('CPG', cpg_solve)
t0 = time.time()
val = prob.solve(method='CPG')
t1 = time.time()
print('\nC solve time:', 1000*(t1-t0), 'ms')
print('C solution: x = ', prob.var_dict['x'].value)
print('C objective function value:', val)
