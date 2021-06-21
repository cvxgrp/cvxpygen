
### Welcome to CVXPYGEN - code generation with CVXPY!

Define a CVXPY problem and generate C code to solve it:

```bash
import cvxpy as cp
import cvxpygen as cpg

# define dimensions, variables, parameters
m, n = 3, 2
x = cp.Variable((n, 1), name='x')
F = cp.Parameter((m, n), name='F')
g = cp.Parameter((m, 1), name='g')
e = cp.Parameter((n, 1), name='e')
delta = cp.Parameter(nonneg=True, name='delta')

# define objective & constraints
objective = cp.sum_squares(F @ x - g) + gamma * cp.sum_squares(x)
constraints = [cp.abs(x) <= e]

# define problem
prob = cp.Problem(cp.Minimize(objective), constraints)

# generate code
cpg.generate_code(prob, code_dir='cpg_code', compile=True)
```

TODO: add more explanations, dependencies, installation instructions, etc.
