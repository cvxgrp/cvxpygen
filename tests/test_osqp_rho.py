import cvxpy as cp
from cvxpygen import cpg


def _initializer(source, declaration):
    start = source.index(declaration)
    return source[start:source.index('};', start)]


def test_osqp_rho_is_baked_into_codegen(tmp_path):
    x = cp.Variable(2)
    q = cp.Parameter(2, value=[1.0, -1.0])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x) + q @ x), [x >= 0])
    code_dir = tmp_path / 'code'
    cpg.generate_code(
        prob,
        code_dir=str(code_dir),
        solver='OSQP',
        solver_opts={'rho': 1.0},
        wrapper=False,
    )
    workspace = (code_dir / 'c' / 'solver_code' / 'workspace.c').read_text()
    rho_inv = _initializer(workspace, 'OSQPFloat linsys_rho_inv_vec[2]')
    assert '10.00000000000000000000' not in rho_inv
    assert rho_inv.count('1.00000000000000000000') == 2
