
# Welcome to CVXPYGEN - code generation with CVXPY!

## Installation

*Note: When the first release is available, installation will consist of a single pip / conda package including all dependencies.*

1. Clone this repository and initialize its submodules.
   ```
   git clone git@github.com:cvxgrp/codegen.git
   git submodule update --init
    ```


2. Install [conda](https://docs.conda.io/en/latest/) and create a new environment,
    ```
    conda create --name cpg_env
    conda activate cpg_env
    ```
    or activate an existing one. Make sure to use the python interpreter of this environment.
   

3. Install ``cvxpy`` and ``CMake``
    ```
   conda install -c conda-forge cvxpy
   conda install -c anaconda cmake
   ```
   
## Example

*Note: The example will be simpler. For development purposes, more variables, parameters etc. are chosen.*

### 1. Generate Code

Define a convex optimization problem the way you are used to with CVXPY.

```python
import cvxpy as cp

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
```

Generating C code for this problem is as simple as,

```python
import gen as cpg  # 'gen' will be 'cvxpygen'
cpg.generate_code(prob, code_dir='CPG_code')
```

where ``code_dir`` specifies the directory that the generated code is stored in.
The above steps are summarized in ``example_main.py``.

To get an overview of the code generation result, have a look at `CPG_code/README.html`.

### 2. Compile Code

To compile the code, you can execute the following in your terminal.

```bash
cd CPG_code/build
cmake ..
make
```

### 3. Solve & Compare

As summarized in ``example_test.py``, you can assign parameter values and solve the problem both by conventional CVXPY and via the generated code, which is wrapped inside the custom CVXPY solve method ``cpg_solve``.

```python
from CPG_code.cpg_solver import cpg_solve
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

# solve problem with C code via python wrapper
prob.register_solve('CPG', cpg_solve)
obj = prob.solve(method='CPG')
print('C result:')
print('f =', obj)
print('x =', prob.var_dict['x'].value)
print('y =', prob.var_dict['y'].value)
```

Observe that both the objective values and solutions are close, comparing python and C results.