
import os
import sys
import cvxpy as cp
import numpy as np

sys.path.append('../')
from cvxpygen import cpg


def _small_qp():
    x = cp.Variable(2, name='x')
    a = cp.Parameter(2, nonneg=True, name='a')
    b = cp.Parameter(name='b')
    prob = cp.Problem(cp.Minimize(a @ (x ** 2)), [cp.sum(x) == b, x >= 0])
    a.value = np.array([1.0, 2.0])
    b.value = 3.0
    return prob


def _read_readme(code_dir):
    path = os.path.join(code_dir, 'README.html')
    assert os.path.isfile(path), f'README.html not found at {path}'
    with open(path) as f:
        return f.read()


def test_readme_no_gradient():
    prob = _small_qp()
    code_dir = 'test_readme_no_grad'
    cpg.generate_code(prob, code_dir=code_dir, solver='QOCO', wrapper=False)

    html = _read_readme(code_dir)

    # No unrendered Jinja2 tags
    assert '{{' not in html
    assert '{%' not in html

    # Problem identity
    assert code_dir in html
    assert 'QOCO' in html

    # Parameters and variables appear in the summary tables
    assert 'a' in html
    assert 'b' in html
    assert 'x' in html

    # Core files present in the tree / panels
    for fname in ('cpg_workspace.h', 'cpg_solve.h', 'cpg_workspace.c',
                  'cpg_solve.c', 'cpg_example.c', 'CMakeLists.txt',
                  'cpg_solver.py', 'setup.py'):
        assert fname in html, f'{fname} missing from README'

    # CVXPY Interface section (no tabs when gradient=False)
    assert 'CVXPY Interface' in html
    assert 'register_solve' in html

    # No gradient content
    assert 'cpg_gradient' not in html
    assert 'CVXPYlayers' not in html


def test_readme_gradient():
    prob = _small_qp()
    code_dir = 'test_readme_grad'
    cpg.generate_code(prob, code_dir=code_dir, solver='OSQP', gradient=True, wrapper=False)

    html = _read_readme(code_dir)

    # No unrendered Jinja2 tags
    assert '{{' not in html
    assert '{%' not in html
    
    # Problem identity
    assert code_dir in html
    assert 'OSQP' in html

    # Gradient files present in the tree / panels
    assert 'cpg_gradient.h' in html
    assert 'cpg_gradient.c' in html

    # Differentiable Interface section with both tabs
    assert 'Python Interface' in html
    assert 'CVXPYlayers' in html
    assert 'custom_method' in html
    assert 'forward' in html
    assert 'backward' in html
