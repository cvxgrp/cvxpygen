
### Welcome to CVXPYGEN - the code generator for CVXPY!

Define a CVXPY problem and generate C code to solve it:

```bash
import cvxpy as cp
import cvxpygen as cpg

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
cpg.generate_code(prob, compile_code=True)
```

TODO: add more explanations, dependencies, installation instructions, etc.
