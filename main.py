
import cvxpy as cp
import gen as cpg  # 'gen' will be 'cvxpygen'
import numpy as np
import os

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

# assign parameter values
np.random.seed(26)
delta.value = np.random.rand()
F.value = np.random.rand(m, n)
g.value = np.random.rand(m, 1)
e.value = np.random.rand(n, 1)

# solve problem conventionally
obj = prob.solve()

# generate code
cpg.generate_code(prob, code_dir='CPG_code', compile=True)

# solve problem via generated code
''''''

print('Python result:')
print('f =', obj)
print('x =', x.value)
print('y =', y.value)

# for development purpose, run example program executable (to be replaced by pybind module)
print('Codegen result:')
os.system('cd ' + os.path.join('CPG_code', 'build') + ' && ./cpg_example')


# TODO: increase code quality
# what is defined as parameter in CVXPY is assumed to generally change between solves in C
# keep in mind: sparsity structure of OSQP matrices, traces back to default values of user parameters
