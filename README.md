
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
   and initialize its submodules.
    ```
    cd codegen
    git submodule update --init
    ```


2. Install [conda](https://docs.conda.io/en/latest/) and create the ``cpg_env`` environment via,
    ```
    conda env create -f environment.yml
    conda activate cpg_env
    ```
    and make sure to use the python interpreter of this environment.
   
3. Install the [GCC compiler](https://gcc.gnu.org) (on Windows, install
``mingw32-gcc-g++`` and ``mingw32-make`` via the [MinGW](https://sourceforge.net/projects/mingw/) Installation Manager).
On Windows, you need to install also [Visual Studio 2017 Build Tools](https://download.visualstudio.microsoft.com/download/pr/3e542575-929e-4297-b6c6-bef34d0ee648/639c868e1219c651793aff537a1d3b77/vs_buildtools.exe).
   
4. *Optional:* If you wish to use the example notebooks located in ``examples/``, register a new kernel spec with Jupyter.
    ```
   conda activate cpg_env
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

# define dimensions, variables, parameters
# IMPORTANT: uniquely specify variable and parameter names for them to be recognized in the generated C code
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
import numpy as np

# assign parameter values and solve
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
cpg.generate_code(prob, code_dir='CPG_code', explicit=True)
```

where ``code_dir`` specifies the directory that the generated code is stored in.
When ``explicit=True``, for-loops are unrolled in the canonicalization code.
The above steps are summarized in the first part of ``example.py``.

To get an overview of the code generation result, have a look at `CPG_code/README.html`.

### 2. Solve & Compare

As summarized in the second part ``example.py``, after assigning parameter values, you can solve the problem both by conventional CVXPY and via the generated code, which is wrapped inside the custom CVXPY solve method ``cpg_solve``.

```python
from CPG_code.cpg_solver import cpg_solve
import numpy as np
import pickle
import time
import os

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
val = prob.solve(method='CPG', updated_params=['A', 'b', 'c'])
t1 = time.time()
print('\nC solve time:', 1000*(t1-t0), 'ms')
print('C solution: x = ', prob.var_dict['x'].value)
print('C objective function value:', val)
```

Comparing python and C results for this example, both the solutions and objective values are almost identical.
In general, there might be differences due to the different step size dynamics of the OSQP solver with or without code generation. 
Especially for smaller problems like this, the new solve method ``'CPG'`` is significantly faster than solving without CVXPYGEN.

### 3. Executable

If you wish to compile the example executable on a Unix platform, please run the following commands in your terminal.

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


## Tests

To run tests, install ``pytest`` via

```bash
conda install pytest
```

and execute:

```bash
cd tests
pytest
```