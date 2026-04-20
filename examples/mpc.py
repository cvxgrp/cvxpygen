import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Predictive Control

    We consider a model predictive control (MPC) problem [1, 2].
    Given a linear time-invariant (LTI) system $A \in \mathbf{R}^{n \times n}, B \in \mathbf{R}^{n \times m}$ with state $x \in \mathbf{R}^n$, we determine an optimal control input $u \in \mathbf{R}^m$ by solving the optimization problem
    \[
    \begin{array}{ll}
    \text{minimize} & x_H^T P x_H + \sum_{t=0}^{H-1} \left(x_t^T Q x_t + u_t^T R u_t \right)\\
    \text{subject to} & x_{t+1} = A x_t + B u_t \quad t = 0,\ldots, H-1, \\
    & \| u_t \|_\infty \leq 1, \quad t = 0,\ldots, H-1, \\
    & x_0 = x_\text{init}, \\
    \end{array}
    \]
    where $x_0, \ldots, x_H$ and $u_0, \ldots, u_{H-1}$ are the variables.
    The prediction horizon is $H$, the control input is constrained within a box of size $1$, and the cost matrices are positive definite, $P, Q, R \succ 0$.
    Usually, $P$ is chosen as the solution to the discrete-time algebraic Riccati equation for the given LTI system and cost matrices $Q, R$.
    The measurement $x_\mathrm{init}$ is assigned to the first in the sequence of states.
    We arrange the state and input variables to matrices $X$ and $U$ whose $t$th column contains $x_t$ or $u_t$, respectively, and reformulate the problem to be [DPP-compliant](https://www.cvxpy.org/tutorial/dpp/index.html) as
    \[
    \begin{array}{ll}
    \text{minimize} & \| P^{1/2} X_{H} \|_2^2 + \| Q^{1/2} X_{0:H-1} \|_F^2 + \| R^{1/2} U \|_F^2\\
    \text{subject to} & X_{1:H} = A X_{0:H-1} + B U, \\
    & | U | \leq \mathbf{1}, \\
    & X_{0} = x_\mathrm{init}, \\
    \end{array}
    \]
    with variables $X \in \mathbf{R}^{n \times H+1}$ and $U \in \mathbf{R}^{m \times H}$.
    Here, $|U|$ denotes the element-wise absolute value of $U$.

    We start by constructing the dynamics matrices $A$ and $B$.
    In this case, the state $x = (p, v)$ consists of position $p \in \mathbf{R}^3$ and velocity $v \in \mathbf{R}^3$ of some rigid body in 3D space.
    The control input $u$ is the force vector that acts on the body's center of mass and rotational dynamics are not considered.
    The discretization step is denoted by $h > 0$.
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
def _(np):
    # dimensions
    H, n, m = 5, 6, 3

    # continuous-time dynamics
    Ac = np.vstack((
        np.hstack((np.zeros((n // 2, n // 2)), np.eye(n // 2))),
        np.zeros((n // 2, n)),
    ))
    Bc = np.vstack((
        np.zeros((n // 2, n // 2)),
        np.eye(n // 2)
    ))  # mass = 1

    # discrete-time dynamics
    h = 0.1
    A = np.eye(n) + h * Ac
    B = h * Bc
    return A, B, H, m, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's define the corresponding CVXPY problem.
    """)
    return


@app.cell
def _(A, B, H, cp, m, n):
    # variables
    X = cp.Variable((n, H + 1), name="X")
    U = cp.Variable((m, H), name="U")

    # parameters
    Proot = cp.Parameter((n, n), name="Proot")
    Qroot = cp.Parameter((n, n), name="Qroot")
    Rroot = cp.Parameter((m, m), name="Rroot")
    xinit = cp.Parameter(n, name="xinit")

    # problem
    obj = cp.Minimize(
        cp.sum_squares(Proot @ X[:, H])
        + cp.sum_squares(Qroot @ X[:, :H])
        + cp.sum_squares(Rroot @ U)
    )
    constr = [
        X[:, 1:] == A @ X[:, :H] + B @ U,
        cp.abs(U) <= 1,
        X[:, 0] == xinit,
    ]
    problem = cp.Problem(obj, constr)
    return Proot, Qroot, Rroot, problem, xinit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code and use the Python interface to register a custom CVXPY method.
    """)
    return


@app.cell
def _(cpg, problem):
    cpg.generate_code(problem, code_dir="mpc_code")
    from mpc_code.cpg_solver import cpg_solve
    problem.register_solve('CPG', cpg_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We assign parameter values and solve the problem. Due to Python overhead, the speed-up with CVXPYgen is moderate.
    """)
    return


@app.cell
def _(Proot, Qroot, Rroot, cp, m, n, np, problem, time, xinit):
    # assign parameter values
    Proot.value = np.eye(n)
    Qroot.value = np.eye(n)
    Rroot.value = np.sqrt(0.1) * np.eye(m)
    xinit.value = np.array([2, 2, 2, -1, -1, 1])

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [1] Wang, Y., and Boyd, S. Fast model predictive control using online optimization. *IEEE Transactions on control systems technology* 18(2), 267-278 (2009)

    [2] Hovgaard, T. G., Larsen, L. F., Jørgensen, J. B., and Boyd, S.  MPC for wind power gradients—utilizing forecasts, rotor inertia, and central energy storage. *2013 European Control Conference (ECC)*, IEEE (2013)
    """)
    return


if __name__ == "__main__":
    app.run()
