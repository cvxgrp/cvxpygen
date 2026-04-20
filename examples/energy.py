import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Energy Management

    We have an electric storage device with state-of-charge (SOC) $q_t \geq 0$ at time $t$ and capacity $Q > 0$.
    We denote the charging power from time $t$ to time $t+1$ as $u_t \in \mathbf{R}$ such that $q_{t+1} = q_t + hu_t$, where $h > 0$ is the length of the time interval.
    The charging power is limited as $-D \leq u_t \leq C$ (where $D, C > 0$).
    The energy price $P(u_t)$ is higher when buying energy from the grid compared to when selling energy to the grid.
    Specifically,
    \[
    P(u_t) =
    \begin{cases}
    p_t u_t (1+\eta) & \text{if} \ u_t \geq 0 \\
    p_t u_t (1-\eta) & \text{otherwise},
    \end{cases}
    \]
    where $p_t \geq 0$ is the average market price at time $t$ and $0 < \eta < 1$.
    To optimize the cost of charging the energy storage from empty to full within a time period of $T$ time steps, we solve the optimization problem
    \[
    \begin{array}{ll}
    \text{minimize} & \sum_{t=0}^{T-1} \left(h p_t \left(u_t + \eta |u_t|\right) + \gamma u_t^2 \right)\\
    \text{subject to} & q_{t+1} = q_t + h u_t, \quad t = 0, \ldots, T-1,\\
    & -D \leq u_t \leq C, \quad t = 0,\ldots,T-1,\\
    & 0 \leq q_t \leq Q, \quad t = 0,\ldots,T-1,\\
    & q_0 = 0, \quad q_T = Q,
    \end{array}
    \]
    where $u_0, \ldots, u_{T-1}$ and $q_0, \ldots, q_T$ are the variables.
    We have added the regularization term $\gamma u_t^2$ to reduce stress on the electronic system due to peak power values, where $\gamma \geq 0$.
    The prices $p_0, \ldots, p_{T-1}$, $\eta$, and $\gamma$ are the parameters.
    The remaining symbols are constants.
    We reformulate the problem to be [DPP-compliant](https://www.cvxpy.org/tutorial/dpp/index.html) by introducing the parameter $s_t = p_t \eta \geq 0$, such that the objective becomes
    $\sum_{t=0}^{T-1} \left(h p_t u_t + h s_t |u_t| + \gamma u_t^2 \right)$.

    Let's define the corresponding CVXPY problem.
    We consider a one-day horizon with a charging decision every 5 minutes, such that $T=24 \cdot 12 = 288$.
    We consider prices in USD/kWh, powers in kW, and $h=1/5$.
    We set $Q = 12$ and $D = C = 2Q/(hT) = 1$.
    Note that we could have declared $Q$, $D$, and $C$ as parameters, too.
    """)
    return


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from cvxpygen import cpg
    from visualization.energy import draw

    return cp, cpg, draw, mo, np, time


@app.cell
def _(cp):
    # dimension and constants
    T = 288
    h = 1 / 5
    Q = 12
    D = 2 * Q / (h * T)
    C = 2 * Q / (h * T)

    # variables
    u = cp.Variable(T, name="u")
    q = cp.Variable(T + 1, name="q")

    # parameters
    p = cp.Parameter(T, nonneg=True, name="p")
    s = cp.Parameter(T, nonneg=True, name="s")
    gamma = cp.Parameter(nonneg=True, name="gamma")

    # problem
    obj = cp.Minimize(h * p @ u + h * s @ cp.abs(u) + gamma * cp.sum_squares(u))
    constr = [
        q[1:] == q[:-1] + h * u,
        -D <= u, u <= C,
        0 <= q, q <= Q,
        q[0] == 0, q[T] == Q,
    ]
    problem = cp.Problem(obj, constr)
    return gamma, p, problem, q, s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code and use the Python interface to register a custom CVXPY method.
    """)
    return


@app.cell
def _(cpg, problem):
    cpg.generate_code(problem, code_dir="charging_code")
    from charging_code.cpg_solver import cpg_solve
    problem.register_solve('CPG', cpg_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We assign parameter values and solve the problem. The one-day period starts and ends at midnight, with a low price level between midnight and 5 AM, and between 8 AM and noon. There is a high price level between 5 AM and 8 AM, and between 5 PM and midnight. Between noon and 5 PM, there is a medium price level (see the visualization below).

    Due to Python overhead, the speed-up with CVXPYgen is moderate.
    """)
    return


@app.cell
def _(cp, gamma, np, p, problem, s, time):
    # assign parameter values
    eta = 0.1
    p.value = np.hstack((np.ones(5 * 12), 5 * np.ones(3 * 12), np.ones(4 * 12), 3 * np.ones(5 * 12), 5 * np.ones(7 * 12)))
    s.value = eta * p.value
    gamma.value = 0.3

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
    print(f'CVXPY\t\t{val_cvxpy:.1f}\t{1e3 * t_cvxpy:.2f} ms')
    print(f'CVXPYgen\t{val_cpg:.1f}\t{1e3 * t_cpg:.2f} ms')
    return


@app.cell
def _(draw, p, q):
    # visualize results
    draw(q.value, p.value)
    return


if __name__ == "__main__":
    app.run()
