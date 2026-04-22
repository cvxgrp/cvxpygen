import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Actuator Allocation

    When controlling dynamical systems like spacecraft or automobiles, the controller commands a desired control input $w \in \mathbf{R}^m$.
    Here, we consider a wrench vector, containing forces and torques in 3D ($m=6$).

    Usually, multiple actuators are available to produce this control input.
    For $n$ actuators, the vector $u \in \mathbf{R}^n$ contains the respective actuation values.
    If $n > m$ and the actuators are in general position, we call the system *over-actuated*, since there are many realizations of $u$ that result in the same value of $w$, via a linear mapping, denoted by $A$.

    Having this freedom of choice, we want to minimize energy consumption, modeled as $\kappa^T | u |$ (with $\kappa \succeq 0$), while discouraging rapid changes of the actuation values, *i.e.*, $\lambda^\mathrm{sm} \Vert u-u^\mathrm{prev} \Vert_2^2$ with $\lambda^\mathrm{sm} \geq 0$ and $u^\mathrm{prev}$ being the actuation of the previous time step.
    Given the bounds $u^\mathrm{min} \preceq u \preceq u^\mathrm{max}$, there might be cases when the desired control input $w$ is infeasible.
    Hence, we only softly penalize deviations between desired and actual control input with the cost term $\Vert A u - w \Vert_2^2$.
    We solve the optimization problem
    \[
    \begin{array}{ll}
    \text{minimize} \quad &\Vert A u - w \Vert_2^2  + \lambda^\mathrm{sm} \Vert u-u^\mathrm{prev} \Vert_2^2 + \kappa^T | u |\\
    \text{subject to} \quad &u^\mathrm{min} \preceq u \preceq u^\mathrm{max},
    \end{array}
    \]
    with variable $u \in \mathbf{R}^n$. The remaining symbols are parameters.
    To make the problem [DPP-compliant](https://www.cvxpy.org/tutorial/dpp/index.html), we introduce the additional variable $\Delta u = u - u^\mathrm{prev}$ and solve
    \[
    \begin{array}{ll}
    \text{minimize} \quad &\Vert A u - w \Vert_2^2  + \lambda^\mathrm{sm} \Vert \Delta u \Vert_2^2 + \kappa^T | u |\\
    \text{subject to} \quad &u^\mathrm{min} \preceq u \preceq u^\mathrm{max}, \\
    &\Delta u = u-u^\mathrm{prev}.
    \end{array}
    \]
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
    from visualization.actuator import draw

    return cp, cpg, draw, mo, np, time


@app.cell
def _(cp):
    # dimensions
    m, n = 3, 8

    # variables
    u       = cp.Variable(n, name="u")
    delta_u = cp.Variable(n, name="delta_u")

    # parameters
    A       = cp.Parameter((m, n), name="A")
    w       = cp.Parameter(m, name="w")
    lamb_sm = cp.Parameter(nonneg=True, name="lamb_sm")
    kappa   = cp.Parameter(n, nonneg=True, name="kappa")
    u_prev  = cp.Parameter(n, name="u_prev")
    u_min   = cp.Parameter(n, name="u_min")
    u_max   = cp.Parameter(n, name="u_max")

    # problem
    obj = cp.Minimize(cp.sum_squares(A @ u - w) + lamb_sm * cp.sum_squares(delta_u) + kappa @ cp.abs(u))
    constr = [u_min <= u, u <= u_max, delta_u == u - u_prev]
    problem = cp.Problem(obj, constr)
    return A, kappa, lamb_sm, n, problem, u_max, u_min, u_prev, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code and use the Python interface to register a custom CVXPY method.
    """)
    return


@app.cell
def _(cpg, problem):
    # generate code and register custom CVXPY solve method
    cpg.generate_code(problem, code_dir="actuator_code")
    from actuator_code.cpg_solver import cpg_solve
    problem.register_solve("CPG", cpg_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We assign parameter values and solve the problem. In this case, the wrench vector $w = (f, t)$ consists of force $f \in \mathbf{R}^2$ and torque $t \in \mathbf{R}$ in two-dimensional space. The control input $u \in \mathbf{R}^8$ contains four pairs of horizontal and vertical forces at 4 positions in the plane (see the visualization below).

    Due to Python overhead, the speed-up with CVXPYgen is moderate.
    """)
    return


@app.cell
def _(A, cp, kappa, lamb_sm, n, np, problem, time, u_max, u_min, u_prev, w):
    # assign parameter values
    A.value = np.array([
        [1,  0, 1, 0,  1, 0,  1,  0],
        [0,  1, 0, 1,  0, 1,  0,  1],
        [1, -1, 1, 1, -1, 1, -1, -1],
    ])
    w.value = np.array([1.0, 1.0, 1.0])
    lamb_sm.value = 0.5
    kappa.value = 0.1 * np.ones(n)
    u_prev.value = np.zeros(n)
    u_min.value = -np.ones(n)
    u_max.value = np.ones(n)

    # time solves
    t = time.time()
    val_cvxpy = problem.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, polish=False)  # match code gen default settings
    t_cvxpy = time.time() - t

    t = time.time()
    val_cpg = problem.solve(method="CPG")
    t_cpg = time.time() - t

    print(f'\t\t\tvalue\ttime')
    print(f'CVXPY\t\t{val_cvxpy:.4f}\t{1e3 * t_cvxpy:.2f} ms')
    print(f'CVXPYgen\t{val_cpg:.4f}\t{1e3 * t_cpg:.2f} ms')
    return


@app.cell(hide_code=True)
def _(mo):
    force_magnitude = mo.ui.slider(
        start=0, stop=1, step=0.01, value=0.75,
        label="Force (Magnitude)"
    )
    force_angle = mo.ui.slider(
        start=0, stop=360, step=1, value=45,
        label="Force (Angle)"
    )
    torque = mo.ui.slider(
        start=-1, stop=1, step=0.01, value=-0.3,
        label="Torque"
    )
    mo.hstack(
        [force_magnitude, force_angle, torque],
        justify="start",
        gap=2,
    )
    return force_angle, force_magnitude, torque


@app.cell
def _(draw, force_angle, force_magnitude, problem, torque):
    draw(problem, force_magnitude.value, force_angle.value, torque.value)
    return


if __name__ == "__main__":
    app.run()
