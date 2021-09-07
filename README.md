
## Welcome to CVXPYGEN - code generation with CVXPY!

### Installation

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
   

3. Install ``cvxpy``
    ```
   conda install -c conda-forge cvxpy
   ```
   

4. Install ``CMake``
    ```
   conda install -c anaconda cmake
    ```
   
### Example

*Note: The example will be simpler. For development purposes, more variables, parameters etc. are chosen.*

Define a convex optimization problem the way you are used to with CVXPY.

```python
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
```

Generating C code for this problem is as simple as,

```python
cpg.generate_code(prob, code_dir='CPG_code')
```

where ``code_dir`` specifies the directory that the generated code is stored in.

To get an overview of the code generation result, have a look at `CPG_code/README.html`.

To compile the code, you can execute the following in your terminal.

```bash
cd CPG_code/build
cmake ..
make
cp cpg_module.cpython-39-darwin.so ../..
```

Assign parameter values and solve the problem both by conventional CVXPY and via the generated code.

```python
# assign parameter values
np.random.seed(26)
delta.value = np.random.rand()
F.value = np.random.rand(m, n)
g.value = np.random.rand(m, 1)
e.value = np.random.rand(n, 1)

# solve problem conventionally
obj = prob.solve()
print('Python result:')
print('f =', obj)
print('x =', x.value)
print('y =', y.value)

# solve problem with C code via python wrapper (to be replaced with custom solve method)
print('C result:')
cpg_module.run_example()
```

Observe that both the objective values and solutions are close, comparing python and C results.

The above steps are summarized in ``main.py``.