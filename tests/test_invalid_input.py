
import pytest
import cvxpy as cp
from cvxpygen import cpg


def test_no_parameters():
    
    # problem
    x = cp.Variable(3, nonneg=True, name='x')
    b = cp.Parameter(1, name='b')
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x) + cp.square(b)))
    
    # assert error when trying to generate code
    with pytest.raises(ValueError, match='Solution does not depend on parameters'):
        cpg.generate_code(prob, solver='OSQP')
