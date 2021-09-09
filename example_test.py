
from CPG_code.cpg_solver import cpg_solve
import numpy as np
import pickle

# load the serialized problem formulation
with open('CPG_code/problem.pickle', 'rb') as f:
    prob = pickle.load(f)

# assign parameter values
np.random.seed(0)
prob.param_dict['A'].value = np.random.randn(3, 2)
prob.param_dict['b'].value = np.random.randn(3,)
prob.param_dict['c'].value = np.random.rand()

# solve problem conventionally
val = prob.solve()
print('Python solution: x = ', prob.var_dict['x'].value)
print('Python objective function value:', val)

# solve problem with C code via python wrapper
prob.register_solve('CPG', cpg_solve)
val = prob.solve(method='CPG')
print('C solution: x = ', prob.var_dict['x'].value)
print('C objective function value:', val)
