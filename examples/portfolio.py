import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Portfolio Construction

    We consider a finanical portfolio optimization problem [1, 2].
    We want to choose portfolio weights for $n$ financial assets (fractions of wealth to invest in the respective assets), denoted by $w\in \mathbf{R}^n$ where $\mathbf{1}^T w = 1$.
    We are given a vector of predicted asset returns $\alpha \in \mathbf{R}^n$ and a predicted asset covariance matrix $\Sigma \succeq 0$.
    For a trade-off between portfolio return and portfolio risk, we use the risk aversion factor $\gamma \geq 0$.
    We also consider (relative) transaction and short-selling cost, denoted by $\kappa_\text{tc} \in \mathbf{R}^n_+$ and $\kappa_\text{sh} \in \mathbf{R}^n_+$, respectively.
    We solve the optimization problem
    \[
    \begin{array}{ll}
    \text{maximize} & \alpha^T w - \gamma w^T \Sigma w - \kappa_\text{tc}^T |w-w^\text{prev}| - \kappa_\text{sh}^T (w)_- \\
    \text{subject to} & \mathbf{1}^T w = 1, \quad \| w \|_1 \leq L,
    \end{array}
    \]
    where $w \in \mathbf{R}^n$ is the variable, $w^\text{prev}$ is the previous portfolio, and $L \geq 1$ is the maximum leverage (both are parameters, along with the remaining symbols).
    The absolute value is taken elementwise and $(x)_- = \max \{-x, 0\}$.

    We use a factor model [3] with $m$ factors such that $\Sigma = F \Sigma^\text{f} F^T + D$ with factor loadings $F \in \mathbf{R}^{n \times m}$, factor covariance $\Sigma^\text{f} \succeq 0$, and idiosyncratic risk $D$ (diagonal).
    We write the [DPP-compliant](https://www.cvxpy.org/tutorial/dpp/index.html) problem
    \[
    \begin{array}{ll}
    \text{maximize}
    & a^T w - \| \left(\Sigma^\text{f}\right)^{1/2} f \|_2^2 - \| D^{1/2} w \|_2^2 - k_\text{tc}^T |\Delta w| - k_\text{sh}^T (w)_- \\
    \text{subject to} & \mathbf{1}^T w = 1, \quad \| w \|_1 \leq L,\\
    & f = F^T w, \quad \Delta w = w-w^\text{prev},\\
    \end{array}
    \]
    where $w, \Delta w \in \mathbf{R}^n$ and $f \in \mathbf{R}^m$ are variables.
    The parameters are
    \[
    a = \alpha / \gamma, \quad F, \quad \left(\Sigma^\text{f}\right)^{1/2}, \quad D^{1/2}, \quad
    k_\text{tc} = \kappa_\text{tc} / \gamma, \quad
    k_\text{sh} = \kappa_\text{sh} / \gamma, \quad
    w^\text{prev}, \quad L.
    \]
    Note that we divided the objective function by the risk aversion factor $\gamma$. This way, updating the value of $\gamma$ only affects the linear part of the objective function, avoiding to compute a matrix factorization when solving the problem repeatedly.

    Let's define the corresponding CVXPY problem. Note that we represent the diagonal matrix $D^{1/2}$ as a vector.
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
    n, m = 100, 10

    # variables
    w = cp.Variable(n, name="w")
    delta_w = cp.Variable(n, name="delta_w")
    f = cp.Variable(m, name="f")

    # parameters
    a = cp.Parameter(n, name="a")
    F = cp.Parameter((n, m), name="F")
    Sigma_f_root = cp.Parameter((m, m), name="Sigma_f_root")
    d_root = cp.Parameter(n, name="d_root")
    k_tc = cp.Parameter(n, nonneg=True, name="k_tc")
    k_sh = cp.Parameter(n, nonneg=True, name="k_sh")
    w_prev = cp.Parameter(n, name="w_prev")
    L = cp.Parameter(name="L")

    # problem
    obj = cp.Maximize(
        a @ w
        - cp.sum_squares(Sigma_f_root @ f)
        - cp.sum_squares(cp.multiply(d_root, w))
        - k_tc @ cp.abs(delta_w)
        - k_sh @ cp.neg(w)
    )
    constr = [
        f == F.T @ w,
        cp.sum(w) == 1,
        cp.norm1(w) <= L,
        delta_w == w - w_prev,
    ]
    problem = cp.Problem(obj, constr)
    return F, L, Sigma_f_root, a, d_root, k_sh, k_tc, m, n, problem, w_prev


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code and use the Python interface to register a custom CVXPY method.
    """)
    return


@app.cell
def _(cpg, problem):
    cpg.generate_code(problem, code_dir="portfolio_code")
    from portfolio_code.cpg_solver import cpg_solve
    problem.register_solve('CPG', cpg_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We assign parameter values and solve the problem. Due to Python overhead, the speed-up with CVXPYgen is moderate.
    """)
    return


@app.cell
def _(
    F,
    L,
    Sigma_f_root,
    a,
    cp,
    d_root,
    k_sh,
    k_tc,
    m,
    n,
    np,
    problem,
    time,
    w_prev,
):
    # assign parameter values
    np.random.seed(0)
    gamma = 1.0
    alpha = np.random.randn(n)
    kappa_tc = 0.001 * np.ones(n)
    kappa_sh = 0.002 * np.ones(n)

    a.value = alpha / gamma
    F.value = np.random.randn(n, m)
    Sigma_f_root.value = np.random.rand(m, m)
    d_root.value = np.random.rand(n)
    k_tc.value = kappa_tc / gamma
    k_sh.value = kappa_sh / gamma
    w_prev.value = np.ones(n) / n
    L.value = 1.6

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
    [1] Lobo, M. S., Fazel, M. and Boyd, S. Portfolio optimization with linear and fixed transaction costs. *Annals of Operations Research* 152(1), 341-365 (2007)

    [2] Moehle, N., Kochenderfer, M. J., Boyd, S., and Ang, A. Tax-aware portfolio construction via convex optimization. *Journal of Optimization Theory and Applications* 189(2), 364-383 (2021)

    [3] Lettau, M. and Pelger, M. Factors that fit the time series and cross-section of stock returns. *The Review of Financial Studies* 33(5), 2274–2325 (2020)
    """)
    return


if __name__ == "__main__":
    app.run()
