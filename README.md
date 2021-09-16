
# Welcome to CVXPYGEN - code generation with CVXPY!

CVXPYGEN takes a convex optimization problem family modeled with CVXPY and generates a corresponding solver in C.
This custom solver is specific to the problem family and accepts different parameter values.
In particular, this solver is suitable for deployment on embedded systems.

CVXPYGEN accepts CVXPY problems that are compliant with [Disciplined Convex Programming (DCP)](https://www.cvxpy.org/tutorial/dcp/index.html).
DCP is a system for constructing mathematical expressions with known curvature from a given library of base functions. 
CVXPY uses DCP to ensure that the specified optimization problems are convex.
In addition, problems need to be modeled according to [Disciplined Parametrized Programming (DPP)](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming).
Solving a DPP-compliant problem repeatedly for different values of the parameters can be much faster than repeatedly solving a new problem.

For now, CVXPYGEN is a separate module, until it will be integrated into CVXPY.
As of today, CVXPYGEN works with linear and quadratic programs.

## Installation

1. Clone this repository via SSH,
    ```
    git clone git@github.com:cvxgrp/codegen.git
    ```
   or via HTTPS.
    ```
    git clone https://github.com/cvxgrp/codegen.git
    ```


2. Install [conda](https://docs.conda.io/en/latest/) and create a new environment,
    ```
    conda create --name cpg_env
    conda activate cpg_env
    ```
    or activate an existing one. Make sure to use the python interpreter of this environment.
   

3. Install ``CVXPY``, ``CMake``, and ``pybind11``
    ```
   conda install -c conda-forge cvxpy
   conda install -c anaconda cmake
   pip install pybind11
   ```
   
4. *Optional:* If you wish to use the example notebooks located in ``examples/``, register a new kernel spec with Jupyter.
    ```
   conda activate cpg_env
   conda install ipykernel
   ipython kernel install --user --name=cpg_env
   ```
   In the Jupyter notebook, click on ``Kernel->Change kernel`` and choose ``cpg_env``.
    
## Example

Let's define a CVXPY problem, generate code for it, and solve the problem with example parameter values.

### 1. Generate Code

Define a convex optimization problem the way you are used to with CVXPY.
Everything that is defined as ``cp.Parameter()`` is assumed to be changing between multiple solves.
For constant properties, use ``cp.Constant()``.

```python
import cvxpy as cp
import numpy as np

# define dimensions, variables, parameters
# IMPORTANT: specify variable and parameter names to recognize them in the generated C code
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
```

Assign parameter values and solve the problem.

```python
# assign parameter values and solve
# IMPORTANT: parameter values must be (reasonably) initialized before generating code, and can be updated later on
np.random.seed(0)
A.value = np.random.randn(m, n)
b.value = np.random.randn(m)
c.value = np.random.rand()
val = prob.solve()
print('Solution: x = ', x.value)
print('Objective function value:', val)
```

Generating C code for this problem is as simple as,

```python
import cvxpygen as cpg
cpg.generate_code(prob, code_dir='CPG_code')
```

where ``code_dir`` specifies the directory that the generated code is stored in.
The above steps are summarized in ``example_main.py``.

To get an overview of the code generation result, have a look at `CPG_code/README.html`.

### 2. Solve & Compare

As summarized in ``example_test.py``, after assigning parameter values, you can solve the problem both by conventional CVXPY and via the generated code, which is wrapped inside the custom CVXPY solve method ``cpg_solve``.

```python
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
val = prob.solve(method='CPG')
t1 = time.time()
print('\nC solve time:', 1000*(t1-t0), 'ms')
print('C solution: x = ', prob.var_dict['x'].value)
print('C objective function value:', val)
```

Comparing python and C results, both the solutions and objective values are almost identical.
For this example, the new solve method ``'CPG'`` is about one order of magnitude faster than solving without CVXPYGEN.

### 3. Executable

If you wish to compile the example executable, please run the following commands in your terminal.

```bash
cd CPG_code/c/build
cmake ..
make
```

To run the compiled program, type

```bash
cd CPG_code/c/build
./cpg_example
```
