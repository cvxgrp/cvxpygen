import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resource Allocation

    Suppose we have to assign $m$ resources to $n$ jobs. The resource allocation is represented as the matrix $X \in \mathbf{R}^{n \times m}$ where $X_{i,j}$ denotes the amount of resource $j$ allocated to job $i$.
    The utility of a given resource allocation $X$ is
    \[
    \textbf{tr} \left( \min \left\{ X W^T, S\right\} \right),
    \]
    with coefficient matrix $W \in \mathbf{R}^{n \times m}$ and saturation $S$.
    The matrix $S$ is diagonal and stores the maximum possible utility for each job.
    We solve the optimization problem
    \[
    \begin{array}{ll}
    \text{maximize} & \textbf{tr} \left( \min \left\{ X W^T, S\right\} \right)\\
    \text{subject to} & X^\text{min} \preceq X \preceq X^\text{max}, \\
    & X^T \mathbf{1} \preceq r,
    \end{array}
    \]
    with variable $X \in \mathbf{R}^{n \times m}$.
    The maximum and minimum amounts of resources to be allocated are denoted by $X^\text{max} \succeq X^\text{min} \succeq 0$, respectively, and $r$ is the vector of available resources.
    The problem is feasible if $\left(X^\text{min}\right)^T \mathbf{1} \preceq r$.

    Let's define the corresponding CVXPY problem.
    """)
    return


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from cvxpygen import cpg
    from visualization.resource import draw

    return cp, cpg, draw, mo, np, time


@app.cell
def _(cp):
    # dimensions
    n, m = 20, 10

    # variable
    X = cp.Variable((n, m), name='X')

    # parameters
    W = cp.Parameter((n, m), name='W')
    S = cp.Parameter((n, n), diag=True, name='S')
    X_min = cp.Parameter((n, m), name='X_min')
    X_max = cp.Parameter((n, m), name='X_max')
    r = cp.Parameter(m, name='r')

    # problem
    obj = cp.Maximize(cp.trace(cp.minimum(X @ W.T, S)))
    constr = [X_min <= X, X<= X_max, cp.sum(X, axis=0) <= r]
    problem = cp.Problem(obj, constr)
    return S, W, X, X_max, X_min, m, n, problem, r


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code and use the Python interface to register a custom CVXPY method.
    """)
    return


@app.cell
def _(cpg, problem):
    cpg.generate_code(problem, code_dir='resource_code')
    from resource_code.cpg_solver import cpg_solve
    problem.register_solve('CPG', cpg_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We assign parameter values and solve the problem. Due to Python overhead, the speed-up with CVXPYgen is moderate.
    """)
    return


@app.cell
def _(S, W, X_max, X_min, cp, m, n, np, problem, r, time):
    # assign parameter values
    np.random.seed(0)
    W.value = 0.1 + 0.9 * np.random.rand(n, m)
    S.value = np.diag(np.random.rand(n))
    X_min.value = np.zeros((n, m))
    X_max.value = np.ones((n, m))
    r.value = 0.5 + 0.5 * np.random.rand(m)

    # time solves
    n_solves = 10
    t = time.time()
    for _ in range(n_solves):
        val_cvxpy = problem.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, polish=False)  # match code gen default settings
    t_cvxpy = (time.time() - t) / n_solves

    t = time.time()
    for _ in range(n_solves):
        val_cpg = problem.solve(method="CPG")
    t_cpg = (time.time() - t) / n_solves

    print(f'\t\t\tvalue\ttime')
    print(f'CVXPY\t\t{val_cvxpy:.2f}\t{1e3 * t_cvxpy:.2f} ms')
    print(f'CVXPYgen\t{val_cpg:.2f}\t{1e3 * t_cpg:.2f} ms')
    return


@app.cell
def _(S, W, X, draw, r):
    draw(W.value, S.value, r.value, X.value)
    return


if __name__ == "__main__":
    app.run()
