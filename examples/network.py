import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network Flow Optimization

    We have a network of $m$ directed edges that support $n$ flows, each of which goes over a fixed path in the graph.
    We denote the flow values as $f \in \mathbf{R}_+^n$.
    The resulting traffic on the $m$ edges is given by $R f$, where $R \in \{0,1\}^{m \times n}$ is the routing matrix, with $R_{ij}=1$ if flow $j$ goes over edge $i$, and 0 otherwise.

    The edges have capacity $c \in \mathbf{R}_+^m$, so we have $Rf \preceq c$.
    The objective is to maximize the total utility, which is $U(f) = \sum_{i=1}^n U_i(f_i)$ where
    \[
    U_i(f_i) =
    \begin{cases}
    -\infty & \text{if} \ f_i < f^\mathrm{min}_i \\
    w_i f_i & \text{if} \ f^\mathrm{min}_i \leq f_i \leq f_i^\mathrm{max} \\
    w_i f_i^\mathrm{max} & \text{otherwise}.
    \end{cases}
    \]
    The first case is the implicit version of the constraint $f^\mathrm{min} \leq f$.
    Except for the case of degenerate networks, the third condition will never hold at optimality.
    Intuitively, when $f_i > f_i^\mathrm{max}$, reducing $f_i$ at not cost, to free capacity for other flows that run through the edges of flow $i$, improves the objective function.
    We encode this information in the constraint $f \leq f^\mathrm{max}$ and rewrite $U$ in vector form to arrive at the optimization problem
    \[
    \begin{array}{ll}
    \text{maximize} & w^T f \\
    \text{subject to} & R f \preceq c, \\
    & f^\mathrm{min} \preceq f \preceq f^\mathrm{max}.
    \end{array}
    \]
    Here, $f \in \mathbf{R}_+^n$ is the variable and $w, c\succeq 0$, $R$, and $f^\mathrm{max} \succeq f^\mathrm{min} \succeq 0$ are parameters.
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
    from visualization.network import init, draw

    return cp, cpg, draw, init, mo, np, time


@app.cell
def _(cp):
    # dimensions
    n, m = 4, 5

    # variable
    f = cp.Variable(n, name="f")

    # parameters
    w = cp.Parameter(n, name="w")  # here, nonneg=True not needed for DPP
    c = cp.Parameter(m, name="c")
    R = cp.Parameter((m, n), name="R")
    f_min = cp.Parameter(n, name="f_min")
    f_max = cp.Parameter(n, name="f_max")

    # problem
    obj = cp.Maximize(w @ f)
    constr = [R @ f <= c, f_min <= f, f <= f_max]
    problem = cp.Problem(obj, constr)
    return R, c, f_max, f_min, m, n, problem, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate code and use the Python interface to register a custom CVXPY method.
    """)
    return


@app.cell
def _(cpg, problem):
    cpg.generate_code(problem, code_dir="network_code")
    from network_code.cpg_solver import cpg_solve
    problem.register_solve('CPG', cpg_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We assign parameter values and solve the problem. Due to Python overhead, the speed-up with CVXPYgen is moderate.
    """)
    return


@app.cell
def _(R, c, cp, f_max, f_min, m, n, np, problem, time, w):
    # assign parameter values
    np.random.seed(0)
    w.value = np.random.rand(n)
    c.value = n * (0.1 + 0.1 * np.random.rand(m))
    R.value = np.round(np.random.rand(m, n))
    f_min.value = np.zeros(n)
    f_max.value = np.ones(n)

    # time solves
    t = time.time()
    val_cvxpy = problem.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, polish=False)  # match code gen default settings
    t_cvxpy = time.time() - t

    t = time.time()
    val_cpg = problem.solve(method="CPG")
    t_cpg = time.time() - t

    print(f'\t\t\tvalue\ttime')
    print(f'CVXPY\t\t{val_cvxpy:.3f}\t{1e3 * t_cvxpy:.2f} ms')
    print(f'CVXPYgen\t{val_cpg:.3f}\t{1e3 * t_cpg:.2f} ms')
    return


@app.cell(hide_code=True)
def _(mo):
    w1 = mo.ui.slider(start=0.8, stop=1.2, step=0.001, value=1.0, label="Weight 1")
    w2 = mo.ui.slider(start=0.8, stop=1.2, step=0.001, value=1.0, label="Weight 2")
    w3 = mo.ui.slider(start=0.8, stop=1.2, step=0.001, value=1.0, label="Weight 3")
    w4 = mo.ui.slider(start=0.8, stop=1.2, step=0.001, value=1.0, label="Weight 4")
    rect1 = mo.Html("""<div style=" width: 32px; height: 24px; background-color: #3498db; border-radius: 8px;"></div>""")
    rect2 = mo.Html("""<div style=" width: 32px; height: 24px; background-color: #e91e8c; border-radius: 8px;"></div>""")
    rect3 = mo.Html("""<div style=" width: 32px; height: 24px; background-color: #f1c40f; border-radius: 8px;"></div>""")
    rect4 = mo.Html("""<div style=" width: 32px; height: 24px; background-color: #2ecc71; border-radius: 8px;"></div>""")
    mo.hstack([rect1, w1, rect2, w2, rect3, w3, rect4, w4], justify="start", gap=1)
    return w1, w2, w3, w4


@app.cell(hide_code=True)
def _(init, problem):
    init(problem)
    return


@app.cell
def _(draw, problem, w1, w2, w3, w4):
    draw(problem, w1.value, w2.value, w3.value, w4.value)
    return


if __name__ == "__main__":
    app.run()
