from types import SimpleNamespace

import cvxpy as cp

import cvxpygen.solvers.osqp as osqp_mod
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


def test_osqp_codegen_setup_accepts_missing_solver_opts(monkeypatch):
    seen = {}

    class FakeOSQP:
        def setup(self, **kwargs):
            seen.update(kwargs)

        def codegen(self, *args, **kwargs):
            return None

    monkeypatch.setattr(osqp_mod.osqp, 'OSQP', FakeOSQP)
    monkeypatch.setattr(osqp_mod.shutil, 'copyfile', lambda *args, **kwargs: None)
    monkeypatch.setattr(osqp_mod.utils, 'read_write_file', lambda *args, **kwargs: None)

    interface = osqp_mod.OSQPInterface.__new__(osqp_mod.OSQPInterface)
    interface._solver_opts = None
    canon = SimpleNamespace(parameter_canon=SimpleNamespace(
        p={'P': 'P', 'q': 'q', 'A': 'A', 'l': 'l', 'u': 'u'},
    ))
    interface.generate_code(None, '.', '.', '.', canon, False, '')
    assert seen == {'P': 'P', 'q': 'q', 'A': 'A', 'l': 'l', 'u': 'u'}
