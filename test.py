
import cpg_module
import pickle
import numpy as np

# load the serialized problem formulation
with open('CPG_code/problem.pickle', 'rb') as f:
    prob = pickle.load(f)

# assign parameter values
np.random.seed(26)
prob.param_dict['delta'].value = np.random.rand()
prob.param_dict['F'].value = np.random.rand(3, 2)
prob.param_dict['g'].value = np.random.rand(3, 1)
prob.param_dict['e'].value = np.random.rand(2, 1)

# solve problem conventionally
obj = prob.solve()
print('Python result:')
print('f =', obj)
print('x =', prob.var_dict['x'].value)
print('y =', prob.var_dict['y'].value)

# solve problem with C code via python wrapper (to be replaced with custom solve method)
print('C result:')
cpg_module.run_example()
