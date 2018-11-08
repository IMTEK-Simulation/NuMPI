
import numpy as np
import scipy.optimize
from LBFGS_Matrix_H import LBFGS
import pytest

from helpers import minimization_problems as mp

@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock])
@pytest.mark.parametrize("n",[10])
def test_minimize_call(Objective,n):

    result = scipy.optimize.minimize(Objective.f,Objective.startpoint(n),jac=Objective.grad,options ={"gtol":1e-6})
    assert result.success

    np.testing.assert_allclose(np.reshape(result.x, (-1,)), np.reshape(Objective.xmin(n), (-1,)), rtol=1e-7)

def test_multiple_tol():
    pass

def test_gtol():
    pass

def test_g2tol():
    pass

def test_ftol():
    pass

def test():
    pass