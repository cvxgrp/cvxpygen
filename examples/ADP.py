import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Approximate Dynamic Programming

    We consider an optimal control problem that is similar to that described in the [MPC example](https://github.com/cvxgrp/codegen/blob/main/examples/MPC.py).
    However, we assume that the system matrices are functions of the state $x \in \mathbf{R}^n$, *i.e.*, we have the nonlinear system $A(x) \in \mathbf{R}^{n \times n}, B(x) \in \mathbf{R}^{n \times m}$ with control input $u \in \mathbf{R}^m$, the variable.
    The dynamics equation $x_{t+1} = A(x_t) x_t + B(x_t) u_t$ would be a nonconvex constraint.
    Therefore, we apply approximate dynamic programming (ADP) [1] by predicting just one time step ahead and approximating the infinite-horizon cost as $\left(A(x) x + B(x) u\right)^T P \left(A(x) x + B(x) u\right)$ with measurement $x \in \mathbf{R}^n$.
    We solve the optimization problem
    \[
    \begin{array}{ll}
    \text{minimize} & \left(A(x) x + B(x) u\right)^T P \left(A(x) x + B(x) u\right) + u^T R u\\
    \text{subject to} & \| u \|_2 \leq 1,
    \end{array}
    \]
    where $u \in \mathbf{R}^m$ is the variable and constrained to be at most length $1$. The cost matrices are positive definite, $P, R \succ 0$. We write the problem [DPP-compliant](https://www.cvxpy.org/tutorial/dpp/index.html) as
    \[
    \begin{array}{ll}
    \text{minimize} & \| F u + g \|^2 + \| R^{1/2} u \|_2^2\\
    \text{subject to} & \| u \|_2 \leq 1,
    \end{array}
    \]
    where the parameters are $F = P^{1/2} B(x)$, $g = P^{1/2} A(x) x$, and $R^{1/2}$. Let's define the corresponding CVXPY problem.
    """)
    return


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from cvxpygen import cpg

    return cp, cpg, mo, np, time


@app.cell
def _(cp):
    # dimensions
    n, m = 6, 3

    # variable and parameters
    u = cp.Variable(m, name="u")
    F = cp.Parameter((n, m), name="F")
    g = cp.Parameter(n, name="g")
    Rroot = cp.Parameter((m, m), name="Rroot")

    # problem
    obj = cp.Minimize(cp.sum_squares(F @ u + g) + cp.sum_squares(Rroot @ u))
    constr = [cp.norm(u) <= 1]
    problem = cp.Problem(obj, constr)
    return F, Rroot, g, m, n, problem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code and use the Python interface to register a custom CVXPY method.
    """)
    return


@app.cell
def _(cpg, problem):
    cpg.generate_code(problem, code_dir="adp_code")
    from adp_code.cpg_solver import cpg_solve
    problem.register_solve('CPG', cpg_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We assign parameter values and solve the problem. In this case, the state $x = (p, v)$ consists of position $p \in \mathbf{R}^3$ and velocity $v \in \mathbf{R}^3$ of a rigid body in 3D space. The control input $u$ represents aerodynamic actuation. The force vector that acts on the body's center of mass is the aerodynamic actuation times the current velocity. Rotational dynamics are not considered. Air resistance forces relate to the squared velocity. The discretization step is denoted by $h > 0$.

    Due to Python overhead, the speed-up with CVXPYgen is moderate.
    """)
    return


@app.cell
def _(F, Rroot, cp, g, m, n, np, problem, time):
    # assign parameter values

    h = 0.1

    def dynamics(x):
        # continuous-time dynamics
        Ac = np.array([
            [0, 0, 0,   1,    0,    0],
            [0, 0, 0,   0,    1,    0],
            [0, 0, 0,   0,    0,    1],
            [0, 0, 0, -x[3],  0,    0],
            [0, 0, 0,   0, -x[4],   0],
            [0, 0, 0,   0,    0, -x[5]],
        ])
        Bc = np.vstack((np.zeros((3, 3)), np.diag(x[3:])))
        # discretize dynamics
        Ad = np.eye(n) + h * Ac
        Bd = h * Bc
        return Ad, Bd

    Proot = np.eye(n)
    x0 = np.array([2, 2, 2, -1, -1, 1])
    A, B = dynamics(x0)

    F.value = Proot @ B
    g.value = Proot @ A @ x0
    Rroot.value = np.sqrt(0.1) * np.eye(m)

    # time solves
    t = time.time()
    val_cvxpy = problem.solve(solver=cp.QOCO)
    t_cvxpy = time.time() - t

    t = time.time()
    val_cpg = problem.solve(method="CPG")
    t_cpg = time.time() - t

    print(f'\t\t\tvalue\ttime')
    print(f'CVXPY\t\t{val_cvxpy:.2f}\t{1e3 * t_cvxpy:.2f} ms')
    print(f'CVXPYgen\t{val_cpg:.2f}\t{1e3 * t_cpg:.2f} ms')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [1] Wang, Y., O'Donoghue, B., and Boyd, S. Approximate dynamic programming via iterated Bellman inequalities. *International Journal of Robust and Nonlinear Control* 25(10), 1472-1496 (2015)
    """)
    return


if __name__ == "__main__":
    app.run()
