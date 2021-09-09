
# Welcome to CVXPYGEN - code generation with CVXPY!

## Installation

*Note: When the first release is available, installation will consist of a single pip / conda package including all dependencies.*

1. Clone this repository and initialize its submodules.
   ```
   git clone git@github.com:cvxgrp/codegen.git
   cd codegen
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
```

Observe that both the objective values and solutions are close, comparing python and C results.