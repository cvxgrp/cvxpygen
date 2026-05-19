import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimal power split

    We consider a variant of the power management problem in [1]. Suppose a nonnegative electric power load $L$ is served by a PV (photovoltaic solar panel) system, a storage battery, and a grid connection.
    We denote the solar power as $s$, the battery power as $b$, and the grid power as $g$, all of which have unit kW.
    These three power sources supply the load, so we have
    \[
    L = s + b + g.
    \]
    The PV power $s$ is nonnegative and limited by the irradiance $R > 0$ (in kW/m²), times the area of the PV system $A > 0$ (in m²), and some unitless efficiency factor, which we assume to be $1$, for simplicity.
    In other words, we have $0 \leq s \leq RA$.
    The battery power satisfies $|b| \leq \gamma Q$, where $Q$ is the battery capacity (in kWh) and $\gamma > 0$ is a known coefficient (in 1/hour).

    The grid power satisfies $g \geq 0$, *i.e.*, we cannot sell power back to the grid.
    The (positive) price of the grid power is $P$ (in USD/kWh), so the grid cost is $Pgh$, where $h$ is the duration of one time period (in hours), over which we hold the power values constant.

    The battery state of charge at the beginning of the time period is denoted by $q$ (in kWh), and satisfies $0 \leq q \leq Q$.
    At the beginning of the next time period the battery charge is $q^+ = q - h b$.  We must have $0 \leq q^+ \leq Q$.

    We take the cost function
    \[
    Pgh + \alpha (q^+-q^\text{tar})^2 + \beta |b|,
    \]
    where $\alpha, \beta > 0$ have appropriate units, and $q^\text{tar}$ is a given target battery charge value.
    To choose the powers we solve the QP
    \[
    \begin{array}{ll}
    \mbox{minimize} & Pgh + \alpha (q^+ - q^\text{tar})^2 + \beta |b|\\
    \mbox{subject to} & L = s + b + g, \\
    & 0 \leq s \leq RA, \quad |b| \leq \gamma Q, \quad g \geq 0, \\
    & q^+ = q - hb, \quad 0 \leq q^+ \leq Q,
    \end{array}
    \]
    where $s$, $b$, $g$, and $q^+$ are the variables, and $P$, $\alpha$, $\beta$, $L$, $R$, $A$, $Q$, $q$, and $q^\text{tar}$ are parameters.
    This problem is not DPP, due to the product of parameters $R$ and $A$, and not compliant with explicit code generation,
    due to a parameter in front of the quadratic part of the objective.
    We introduce the auxiliary parameter $S = RA$ and divide the objective by $\alpha$, to arrive at the problem
    \[
    \begin{array}{ll}
    \mbox{minimize} & (P / \alpha) gh + (q^+ - q^\text{tar})^2 + (\beta / \alpha) |b|\\
    \mbox{subject to} & L = s + b + g, \\
    & 0 \leq s \leq S, \quad |b| \leq \gamma Q, \quad g \geq 0, \\
    & q^+ = q - hb, \quad 0 \leq q^+ \leq Q,
    \end{array}
    \]
    where $s$, $b$, $g$, and $q^+$ are the variables, and $P / \alpha$, $\beta / \alpha$, $L$, $S$, $Q$, $q$, and $q^\text{tar}$ are parameters.
    We set $\gamma = 1/5$ (it takes 5 hours to fully charge the battery), and $h = 1 / 2$ (corresponding to 30-minute intervals).

    Here is the CVXPY problem.
    """)
    return


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from cvxpygen import cpg
    from utils.power import profiles, plot
    from utils.pgd import pgd

    return cp, cpg, mo, np, pgd, plot, profiles, time


@app.cell
def _(cp):
    # constants
    gamma = 1 / 5
    h = 1 / 2

    # variables
    s = cp.Variable(name='s')
    b = cp.Variable(name='b')
    g = cp.Variable(name='g')
    qplus = cp.Variable(name='qplus')

    # parameters
    P_over_alpha = cp.Parameter(name='P_over_alpha')
    beta_over_alpha = cp.Parameter(name='beta_over_alpha', nonneg=True)
    L = cp.Parameter(name='L')
    S = cp.Parameter(name='S')
    Q = cp.Parameter(name='Q')
    q = cp.Parameter(name='q')
    qtar = cp.Parameter(name='qtar')

    # problem
    obj = cp.Minimize(P_over_alpha * g * h + (qplus - qtar)**2 + beta_over_alpha * cp.abs(b))
    constr = [
        L == s + b + g,
        0 <= s, s <= S, cp.abs(b) <= gamma * Q, g >= 0,
        qplus == q - h * b, 0 <= qplus, qplus <= Q
    ]
    problem = cp.Problem(obj, constr)
    return (
        L,
        P_over_alpha,
        Q,
        S,
        b,
        beta_over_alpha,
        constr,
        g,
        h,
        obj,
        problem,
        q,
        qplus,
        qtar,
        s,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code twice, with and without an explicit solver.
    For explicit code generation, we use the following limits on parameter values:
    \[
    0.25 \leq P \leq 0.5, \quad
    0.01 \leq \alpha \leq 10, \quad
    0.01 \leq \beta \leq 10, \quad
    0 \leq L \leq 10, \quad 0 \leq S \leq 100, \quad 0 \leq Q \leq 50, \quad 0 \leq q \leq 50.
    \]
    We always set $q^\text{tar} = 0.8 Q$, such that $0 \leq q^\text{tar} \leq 40$.
    """)
    return


@app.cell
def _(
    L,
    P_over_alpha,
    Q,
    S,
    beta_over_alpha,
    constr,
    cp,
    cpg,
    obj,
    problem,
    q,
    qtar,
):
    # implicit solver
    cpg.generate_code(problem, code_dir="power_tuning", gradient=True)

    # explicit solver
    plimits = [
        0.25 / 10 <= P_over_alpha, P_over_alpha <= 0.5 / 0.01,
        0.01 / 10 <= beta_over_alpha, beta_over_alpha <= 10 / 0.01,
        0 <= L, L <= 10,
        0 <= S, S <= 100,
        0 <= Q, Q <= 50,
        0 <= q, q <= 50,
        0 <= qtar, qtar <= 40
    ]
    problem_ex = cp.Problem(obj, constr + plimits)
    cpg.generate_code(problem_ex, code_dir="power_tuning_ex", prefix='ex', solver='explicit', gradient=True)
    return (problem_ex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We construct a default and two custom torch layers with CVXPYlayers.
    """)
    return


@app.cell
def _(
    L,
    P_over_alpha,
    Q,
    S,
    b,
    beta_over_alpha,
    g,
    problem,
    problem_ex,
    q,
    qplus,
    qtar,
    s,
):
    # construct layers
    import torch
    from cvxpylayers.torch import CvxpyLayer
    from cvxpylayers.interfaces import SolverInterface as SI
    from power_tuning.cpg_solver import forward, backward
    from power_tuning_ex.cpg_solver import forward as forward_ex, backward as backward_ex
    params = [P_over_alpha, beta_over_alpha, L, S, Q, q, qtar]
    vars = [s, b, g, qplus]
    layer_ref = CvxpyLayer(problem, parameters=params, variables=vars)
    layer_im = CvxpyLayer(problem, parameters=params, variables=vars, solver=SI.from_codegen(forward, backward))
    layer_ex = CvxpyLayer(problem_ex, parameters=params, variables=vars, solver=SI.from_codegen(forward_ex, backward_ex))
    return layer_ex, layer_im, layer_ref, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will tune $A$ and $Q$ to minimize the cost of purchasing power from the grid, plus the pro-rated cost of acquiring and maintaining a PV system with area $A$ and a battery of size $Q$.
    In particular, we want to minimize
    \[
    \sum_{t=1}^T (P_t g_t h) + (Th/D^\text{life}) (\kappa_A A + \kappa_Q Q),
    \]
    where subscript $t$ denotes the time period, $D^\text{life}$ is the lifetime (in hours) of the PV system and battery (assumed equal, for simplicity), and $\kappa_A, \kappa_Q > 0$ are their prices (in USD/m² and USD/kWh, respectively).

    We assume a price profile $P_1, \ldots, P_T$, where the price is at 0.50 USD/kWh between 4pm and 11pm, and at 0.25 USD/kWh otherwise.
    We assume a load profile $L_1, \ldots, L_T$ with a moderate peak in the morning and a large peak in the afternoon/evening.
    Ultimately, we assume an irradiance profile $R_1, \ldots, R_T$ which corresponds to the irradiance in the San Francisco Bay Area on June 21 with clear sky.

    We perform the design optimization over a week, such that $T = 168 / h = 336$.
    We set $D^\text{life} = 3 \cdot 10^5$ (about 35 years), $\kappa_A = 150$, and $\kappa_Q = 100$.
    """)
    return


@app.cell
def _(plot, profiles, torch):
    # load and price profiles
    T = 336

    P_np, L_np, R_np = profiles()
    plot(P_np, title='Grid price', label='USD/kWh')
    plot(L_np, title='Power load', label='kW')
    plot(R_np, title='Irradiance', label='kW/m²')

    P_th = torch.tensor(P_np)
    L_th = torch.tensor(L_np)
    R_th = torch.tensor(R_np)

    D_life = 3e5
    kappa_A = 150
    kappa_Q = 200
    return D_life, L_th, P_th, R_th, T, kappa_A, kappa_Q


@app.cell
def _(
    D_life,
    L_th,
    P_th,
    R_th,
    T,
    h,
    kappa_A,
    kappa_Q,
    layer_ex,
    layer_im,
    layer_ref,
    np,
    torch,
):
    # simulation
    def simulate(theta, lyr, solver_args={}, compute_grad=True):

        A_th = torch.tensor(theta[0], dtype=torch.float64, requires_grad=True)
        Q_th = torch.tensor(theta[1], dtype=torch.float64, requires_grad=True)
        alpha_th = torch.tensor(theta[2], dtype=torch.float64, requires_grad=True)
        beta_th = torch.tensor(theta[3], dtype=torch.float64, requires_grad=True)

        q_th = [torch.tensor(0.0)]
        g_th = []

        for t in range(T):
            _, _, g, qplus, = lyr(
                P_th[t] / alpha_th,
                beta_th / alpha_th,
                L_th[t],
                R_th[t] * A_th,
                Q_th,
                q_th[-1],
                0.6 * Q_th,
                solver_args=solver_args
            )
            g_th.append(g)
            q_th.append(torch.clamp(qplus, min=torch.tensor(0.0, dtype=torch.float64)))

        # compute cost
        cost = torch.stack(g_th) @ P_th * h + (T * h / D_life) * (kappa_A * A_th + kappa_Q * Q_th)

        if compute_grad:
            cost.backward()
            return cost.item(), np.array([A_th.grad.numpy(), Q_th.grad.numpy(), alpha_th.grad.numpy(), beta_th.grad.numpy()])
        else:
            return cost.item(), np.zeros(2)

    def sim_ref(theta, compute_grad):
        return simulate(theta, layer_ref, solver_args={'eps_abs': 1e-5, 'eps_rel': 1e-5}, compute_grad=compute_grad)

    def sim_im(theta, compute_grad):
        return simulate(theta, layer_im, solver_args={}, compute_grad=compute_grad)

    def sim_ex(theta, compute_grad):
        return simulate(theta, layer_ex, solver_args={}, compute_grad=compute_grad)

    return sim_ex, sim_im, sim_ref


@app.cell
def _(np, pgd, sim_ex, sim_im, sim_ref, time):
    # initialization and parameter limits
    theta_init = np.array([50.0, 25.0, 1.0, 1.0])  # A, Q, alpha, beta
    theta_lower = np.array([0.0, 0.0, 0.01, 0.01])
    theta_upper = np.array([100.0, 50.0, 10.0, 10.0])

    # tune with reference layer
    t_ref = time.time()
    sol_ref, perf_ref, _ = pgd(sim_ref, theta_init, theta_lower, theta_upper, stepsize=0.1, n_iter=25)
    t_ref = time.time() - t_ref

    # tune with custom osqp layer
    t_im = time.time()
    sol_im, perf_im, _ = pgd(sim_im, theta_init, theta_lower, theta_upper, stepsize=0.1, n_iter=25)
    t_im = time.time() - t_im

    # tune with custom explicit layer
    t_ex = time.time()
    sol_ex, perf_ex, _ = pgd(sim_ex, theta_init, theta_lower, theta_upper, stepsize=0.1, n_iter=25)
    t_ex = time.time() - t_ex
    return perf_ex, sol_ex, sol_im, sol_ref, t_ex, t_im, t_ref


@app.cell
def _(sol_ex, sol_im, sol_ref, t_ex, t_im, t_ref):
    print(f'\t\t\t\t\tA\t\tQ\t\ttime')
    print(f'CVXPYlayers\t\t\t{sol_ref[0]:.1f}\t{sol_ref[1]:.3f}\t{t_ref:.2f} s')
    print(f'CVXPYgen OSQP\t\t{sol_im[0]:.1f}\t{sol_im[1]:.3f}\t{t_im:.2f} s')
    print(f'CVXPYgen explicit\t{sol_ex[0]:.1f}\t{sol_ex[1]:.3f}\t{t_ex:.2f} s')
    return


@app.cell
def _(perf_ex):
    # plot convergence
    import matplotlib.pyplot as plt
    plt.plot(perf_ex)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [1] Schaller, M., Arnström, D., Bemporad, A. and Boyd, S. Automatic generation of explicit quadratic programming solvers. To appear, *IEEE Control Systems Magazine* (2026)
    """)
    return


if __name__ == "__main__":
    app.run()
