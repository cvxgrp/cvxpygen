
import platform
import warnings
import pytest
import cvxpy as cp
from cvxpygen import cpg


def test_clarabel():
    
    # trivial problem
    x = cp.Variable(3, nonneg=True, name='x')
    b = cp.Parameter(3, name='b')
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    
    # assert error when trying to solve on Windows
    if platform.system() == 'Windows':
        with pytest.raises(ValueError, match='CLARABEL solver currently not supported on Windows.'):
            cpg.generate_code(prob, solver='CLARABEL')
    else:
        # raise warning and continue code
        warnings.warn('Did not test unsupported solver handling since this is not a Windows machine.')
