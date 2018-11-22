
import numpy as np
import scipy.optimize
import pytest

import tests.minimization_problems as mp

@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock])
@pytest.mark.parametrize("n",[10])
def test_minimize_call(Objective,n):

    ## Test column Vector call
    result = scipy.optimize.minimize(Objective.f,Objective.startpoint(n),jac=Objective.grad,options ={"gtol":1e-6})
    assert result.success
    np.testing.assert_allclose(np.reshape(result.x, (-1,)), np.reshape(Objective.xmin(n), (-1,)), rtol=1e-7)

    ## Test row Vectors call (like scipy and PyCo)
    result = scipy.optimize.minimize(Objective.f, Objective.startpoint(n).reshape(-1), jac=lambda x: Objective.grad(x).reshape(-1), options={"gtol": 1e-6})
    assert result.success, ""
    np.testing.assert_allclose(np.reshape(result.x, (-1,)), np.reshape(Objective.xmin(n), (-1,)), rtol=1e-7)

# TODO: test when jac is Bool

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