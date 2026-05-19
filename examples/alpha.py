import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning a linear model for financial returns

    We construct a portfolio of holdings in $N$ financial assets, with estimated returns $\alpha \in \mathbf{R}^N$.
    We estimate the returns as $\alpha = \Theta x$, where $x \in \mathbf{R}^d$ is a feature vector and the coefficient matrix $\Theta \in \mathbf{R}^{N \times d}$ is to be learned.

    Given an estimate for the covariance of asset returns $\Sigma \in \mathbf{S}^N_{++}$, we choose portfolio weights $w \in \mathbf{R}^N$ by solving the problem
    \[
    \begin{array}{ll}
    \text{maximize} & \alpha^T w - \gamma w^T \Sigma w \\
    \text{subject to} & \mathbf{1}^T w = 1, \quad w \succeq 0,
    \end{array}
    \]
    where $\gamma > 0$ is the risk-aversion factor.

    Following [1], we take $\Sigma$ as the annualized sample covariance of the daily returns of the $N=7$ stocks AAPL, AMZN, BRK.A, FB (now META), GOOGL, MSFT, and XOM, over the years 2017 and 2018, and set $\gamma = 2$.
    """)
    return


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import cvxpy as cp
    from cvxpygen import cpg
    from utils.pgd import pgd

    return cp, cpg, mo, np, pd, pgd, time


@app.cell
def _(pd):
    # load prices, parsing the index as dates
    prices_df = pd.read_csv('7_stocks_17_18_19.csv', index_col=0, header=None, parse_dates=True)

    # count trading days per year
    days_per_year = prices_df.groupby(prices_df.index.year).size()
    T17, T18, T19 = days_per_year.loc[2017], days_per_year.loc[2018], days_per_year.loc[2019]

    # compute returns and their covariance over 2017/18
    returns = prices_df.pct_change().dropna().to_numpy()
    returns_17_18 = returns[:T17+T18]
    returns_19 = returns[-T19:]
    Sigma = 250 * (returns_17_18.T @ returns_17_18 / (T17+T18-1))

    # risk-aversion factor
    gamma = 2
    return Sigma, T19, gamma, returns_19


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is the CVXPY problem specification.
    """)
    return


@app.cell
def _(Sigma, cp, gamma):
    # dimension
    N = 7

    # parameter and variable
    alpha = cp.Parameter(N, name='alpha')
    w = cp.Variable(N, name='w')

    # problem
    obj = cp.Maximize(alpha @ w - gamma * cp.quad_form(w, Sigma))
    constr = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(obj, constr)
    return N, alpha, constr, obj, problem, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code twice, with and without an explicit solver.
    For explicit code generation, we limit the daily return estimate $\alpha$ to be within $\pm100\%$:
    \[
    -1 \preceq \alpha \preceq 1
    \]
    """)
    return


@app.cell
def _(alpha, constr, cp, cpg, obj, problem):
    # implicit solver
    cpg.generate_code(problem, code_dir="alpha", gradient=True)

    # explicit solver
    problem_ex = cp.Problem(obj, constr + [-1 <= alpha, alpha <= 1])
    cpg.generate_code(problem_ex, code_dir="alpha_ex", prefix='ex', solver='explicit', gradient=True)
    return (problem_ex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We construct a default and two custom torch layers with CVXPYlayers.
    """)
    return


@app.cell
def _(alpha, problem, problem_ex, w):
    # construct layers
    import torch
    from cvxpylayers.torch import CvxpyLayer
    from cvxpylayers.interfaces import SolverInterface as SI
    from alpha.cpg_solver import forward, backward
    from alpha_ex.cpg_solver import forward as forward_ex, backward as backward_ex
    params, vars = [alpha], [w]
    layer_ref = CvxpyLayer(problem, parameters=params, variables=vars)
    layer_im = CvxpyLayer(problem, parameters=params, variables=vars, solver=SI.from_codegen(forward, backward))
    layer_ex = CvxpyLayer(problem_ex, parameters=params, variables=vars, solver=SI.from_codegen(forward_ex, backward_ex))
    return layer_ex, layer_im, layer_ref, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the sake of this demo, we simply take the previous days' returns as the features $x$.
    We back-test the portfolio throughout 2019 where the value $V_t$ of the portfolio at time $t$ propagates as
    \[
    V_{t+1} = V_t ((1 + r_t)^T w^\star - \kappa \|w^\star - w^\text{pre}\|_1).
    \]
    Here, $r_t$ is the return from time $t$ to time $t+1$, $w^\star$ is the solution to the portfolio construction problem, $\kappa$ is the relative transaction cost, and $w^\text{pre}$ is the pre-trade portfolio.
    We set $\kappa$ to four basis points and the pre-trade portfolio is updated as
    \[
    w^\text{pre} = w^\star \circ (1 + r_t) V_t / V_{t+1}.
    \]
    The portfolio returns are $R_t = V_t/V_{t-1} - 1$. We denote their mean and standard deviation by $m_R$ and $s_R$, respectively.
    Ultimately, we want to maximize the annualized Sharpe ratio
    \[
    \sqrt{250} (m_R / s_R).
    \]
    """)
    return


@app.cell
def _(N, T19, layer_ex, layer_im, layer_ref, np, returns_19, torch):
    # returns and features
    r_th = 250 * torch.tensor(returns_19[1:])
    x_th = 250 * torch.tensor(returns_19[:-1])

    # relative transaction cost
    kappa = 4e-4

    # backtest
    def backtest(Theta, lyr, solver_args={}, compute_grad=True):

        Theta_th = torch.tensor(Theta, requires_grad=True)
        alpha_th = x_th @ Theta_th.T

        wpre_th = [torch.tensor(np.ones(N) / N)]
        w_th = []
        V_th = [torch.tensor(1.)]

        for t in range(T19 - 1):
            alpha_th_clipped = torch.clip(alpha_th[t], -1.0, 1.0)
            wstar, = lyr(alpha_th_clipped, solver_args=solver_args)
            w_th.append(wstar)
            V_th.append(V_th[-1] * (1 + r_th[t] @ wstar - kappa * torch.norm(wstar - wpre_th[-1], 1)))
            wpre_th.append(wstar * (1 + r_th[t]) * V_th[-2] / V_th[-1])

        # compute Sharpe ratio
        portfolio_values = torch.stack(V_th)
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        mean_return = portfolio_returns.mean()
        std_return = torch.clip(portfolio_returns.std(), 1e-6, None)
        neg_sharpe = -torch.tensor(np.sqrt(250)) * mean_return / std_return

        if compute_grad:
            neg_sharpe.backward()
            return neg_sharpe.item(), Theta_th.grad.numpy()
        else:
            return neg_sharpe.item(), None

    def bt_ref(Theta, compute_grad):
        return backtest(Theta, layer_ref, solver_args={'eps_abs': 1e-5, 'eps_rel': 1e-5}, compute_grad=compute_grad)

    def bt_im(Theta, compute_grad):
        return backtest(Theta, layer_im, solver_args={}, compute_grad=compute_grad)

    def bt_ex(Theta, compute_grad):
        return backtest(Theta, layer_ex, solver_args={}, compute_grad=compute_grad)

    return bt_ex, bt_im, bt_ref


@app.cell
def _(N, bt_ex, bt_im, bt_ref, np, pgd, time):
    # initialization
    np.random.seed(1)
    Theta_init = 0.01 * np.random.randn(N, N)

    # tune with reference layer
    t_ref = time.time()
    sol_ref, perf_ref, _ = pgd(bt_ref, Theta_init, stepsize=0.01, n_iter=30)
    t_ref = time.time() - t_ref

    # tune with custom osqp layer
    t_im = time.time()
    sol_im, perf_im, _ = pgd(bt_im, Theta_init, stepsize=0.01, n_iter=30)
    t_im = time.time() - t_im

    # tune with custom explicit layer
    t_ex = time.time()
    sol_ex, perf_ex, _ = pgd(bt_ex, Theta_init, stepsize=0.01, n_iter=30)
    t_ex = time.time() - t_ex
    return perf_ex, t_ex, t_im, t_ref


@app.cell
def _(t_ex, t_im, t_ref):
    print(f'\t\t\t\t\ttime')
    print(f'CVXPYlayers\t\t\t{t_ref:.2f} s')
    print(f'CVXPYgen OSQP\t\t{t_im:.2f} s')
    print(f'CVXPYgen explicit\t{t_ex:.2f} s')
    return


@app.cell
def _(perf_ex):
    # plot convergence
    import matplotlib.pyplot as plt
    plt.plot(-perf_ex)
    plt.xlabel('Iteration')
    plt.ylabel('Sharpe ratio')
    plt.title('Convergence')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The large Sharpe ratios are to be taken with a grain of salt, as within the scope of this demo, we do not prevent overfitting to the year 2019.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [1] Schaller, M., Arnström, D., Bemporad, A. and Boyd, S. Automatic generation of explicit quadratic programming solvers. To appear, *IEEE Control Systems Magazine* (2026)
    """)
    return


if __name__ == "__main__":
    app.run()
