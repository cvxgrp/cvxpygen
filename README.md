# CVXPYgen

[![PyPI version](https://badge.fury.io/py/cvxpygen.svg)](https://badge.fury.io/py/cvxpygen)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![CI](https://github.com/cvxgrp/cvxpygen/actions/workflows/CI.yml/badge.svg)](https://github.com/cvxgrp/cvxpygen/actions/workflows/CI.yml)

CVXPYgen takes a parametrized [CVXPY](https://www.cvxpy.org/) convex optimization problem and generates a custom C solver tailored to that problem family. The generated code provides interfaces in C/C++ (for embedded applications) and Python/CVXPY (for prototyping and desktop applications).

- **Problem types:** LP, QP, SOCP
- **Solvers:** OSQP, QOCOGEN, CLARABEL, SCS, ECOS, PDAQP
- **Extras:** [differentiable QPs](#differentiable-qps), [explicit solutions for QPs](#explicit-solutions-for-qps)

Problems need to be modeled according to [disciplined parametrized programming (DPP)](https://www.cvxpy.org/tutorial/dpp/index.html). For more details, see our [slides and manuscript](https://web.stanford.edu/~boyd/papers/cvxpygen.html). If you use this code, please cite
```
@article{schaller2022embedded,
  title={Embedded code generation with CVXPY},
  author={Schaller, Maximilian and Banjac, Goran and Diamond, Steven and Agrawal, Akshay and Stellato, Bartolomeo and Boyd, Stephen},
  journal={IEEE Control Systems Letters},
  volume={6},
  pages={2653--2658},
  year={2022},
  publisher={IEEE}
}
```

## Install

```
pip install cvxpygen
```

The CLARABEL solver requires [Rust](https://www.rust-lang.org/tools/install) and [Eigen](https://github.com/oxfordcontrol/Clarabel.cpp#installation). Example notebooks require `marimo` and `matplotlib`.

## Quick start: Nonnegative least squares
```python
import cvxpy as cp
from cvxpygen import cpg

# define CVXPY problem
m, n = 3, 2
A = cp.Parameter((m, n), name='A')
b = cp.Parameter(m, name='b')
x = cp.Variable(n, name='x')
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

# generate code
cpg.generate_code(problem, code_dir='nonneg_ls')
```

**Important:** Set `name=` on variables and parameters. You'll identify them with these names after code generation.
An extended version of this example is available under [quick_start.py](quick_start.py).
An HTML summary of the generated code is written to `nonneg_ls/README.html`, with instructions for using the generated code.

<details>
<summary>C interface</summary>

Here is how to compile the auto-generated example `nonneg_ls/c/src/cpg_example.c` (requires CMake). Parameters and variables are stored as Fortran-order (column-major) flat arrays in C.

**Unix:**
```bash
cd nonneg_ls/c/build
cmake ..
cmake --build . --target cpg_example
./cpg_example
```

**Windows:**
```bash
cd nonneg_ls\c\build
cmake ..
cmake --build . --target cpg_example --config release
Release\cpg_example
```

</details>

<details>
<summary>CVXPY interface</summary>
Here is how to register the Python module as a custom method for CVXPY.

```python
from nonneg_ls.cpg_solver import cpg_solve
problem.register_solve('CPG', cpg_solve)

b.value = ...
problem.solve(method='CPG', updated_params=['b'])

solution = x.value
```
You may pass `updated_params` to tell the solver which parameters changed (omit it to update all).
</details>

## `generate_code` options

| Argument | Type | Default | Description |
|---|---|---|---|
| `code_dir` | `str` | `'cpg_code'` | Output directory |
| `solver` | `str` | `'OSQP'` for QPs <br>`'QOCOGEN'` for SOCPs | Solver backend |
| `solver_opts` | `dict` | `{}` | Options forwarded to the solver |
| `enable_settings` | `list[str]` | `[]` | Solver settings to keep tuneable at runtime |
| `prefix` | `str` | `''` | Symbol prefix for multi-problem codegen |
| `gradient` | `bool` | `False` | Enable gradient computation (QPs only) |
| `wrapper` | `bool` | `True` | Compile Python wrapper |

## Examples

| Domain | Marimo notebook |
|---| --|
| Actuator allocation | [actuator.py](examples/actuator.py) |
| Approximate dynamic programming | [adp.py](examples/adp.py) |
| Model predictive control | [mpc.py](examples/mpc.py) |
| Portfolio construction | [portfolio.py](examples/portfolio.py) |
| Resource allocation | [resource.py](examples/resource.py) |
| Energy management | [energy.py](examples/energy.py) |
| Network flow optimization | [network.py](examples/network.py) |

Run any marimo notebook as:
```
marimo edit actuator.py
```

## Differentiable QPs

CVXPYgen supports differentiating through quadratic programs, with an interface to [CVXPYlayers](https://github.com/cvxgrp/cvxpylayers) (≤ 0.1.9):

```python
cpg.generate_code(problem, code_dir='nonneg_ls_diff', gradient=True)

from nonneg_ls_diff.cpg_solver import forward, backward
from cvxpylayers.torch import CvxpyLayer

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x],
                   custom_method=(forward, backward))
```

As long as the problem is a QP, also conic solvers such as QOCOGEN or CLARABEL are supported.
For more details, see our [manuscript on differentiable QPs](https://stanford.edu/~boyd/papers/cvxpygen_grad.html) and [examples/paper_grad](examples/paper_grad) for examples from the manuscript. If you use this feature, please cite
```
@article{schaller2025code,
  title={Code generation for solving and differentiating through convex optimization problems},
  author={Schaller, Maximilian and Boyd, Stephen},
  journal={Optimization and Engineering},
  pages={1--25},
  year={2025},
  publisher={Springer}
}
```

## Explicit solutions for QPs

For QPs whose parameters are all vectors (no matrix parameters), the solution is a piecewise affine function of the parameters.
CVXPYgen integrates with [PDAQP](https://github.com/darnstrom/pdaqp) to precompute this map offline and generates code for instant evaluation online.
Advantages of such an explicit solver can include transparency, interpretability, reliability, and speed.
The number of inequality constraints (and non-differentiable atoms such as `cp.norm1`) should be small, since the number of affine pieces typically grows exponentially with those.

**Important:** Limit parameters with standard CVXPY constraints to keep the number of affine pieces tractable. For example, specify `l <= p, p <= u` where `p` is a CVXPY parameter and `l` and `u` are arrays:

```python
prob = cp.Problem(obj, constr + [l <= p, p <= u])
cpg.generate_code(prob, code_dir='code_explicit', solver='explicit')
```

To enable the computation of dual variables, set `solver_opts={'dual': True}`.
Control complexity via `'max_floats'` (number of floating point numbers in the explicit solution, default `1e6`) and `'max_regions'` (maximum number of affine pieces, default `500`) in `solver_opts`.
Store the explicit solution in half precision (instead of single precision) by setting `'fp16': True` in `solver_opts`.

For more details, see our [slides and manuscript on explicit solutions for QPs](https://stanford.edu/~boyd/papers/cvxpygen_mpqp.html). If you use this feature, please cite
```
@article{schaller2025automatic,
  title={Automatic generation of explicit quadratic programming solvers},
  author={Schaller, Maximilian and Arnstr{\"o}m, Daniel and Bemporad, Alberto and Boyd, Stephen},
  journal={arXiv preprint arXiv:2506.11513},
  year={2025}
}
```
CVXPYgen 1.0 supports `gradient=True` along with `solver='explicit'`.

## Tests

```bash
pip install pytest
cd tests
pytest
```

## License

[Apache 2.0](LICENSE)
