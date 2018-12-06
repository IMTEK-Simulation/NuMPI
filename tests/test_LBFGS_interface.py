
import numpy as np
import scipy.optimize
import pytest
from MPITools.Optimization import LBFGS

import tests.minimization_problems as mp

@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock])
@pytest.mark.parametrize("n",[10])
def test_minimize_call(Objective,n):

    ## Test column Vector call
    result = scipy.optimize.minimize(Objective.f,Objective.startpoint(n),jac=Objective.grad,method=LBFGS,options ={"gtol":1e-6})
    assert result.success
    np.testing.assert_allclose(np.reshape(result.x, (-1,)), np.reshape(Objective.xmin(n), (-1,)), rtol=1e-7)

    ## Test row Vectors call (like scipy and PyCo)
    result = scipy.optimize.minimize(Objective.f, Objective.startpoint(n).reshape(-1),method=LBFGS, jac=lambda x: Objective.grad(x).reshape(-1), options={"gtol": 1e-6})
    assert result.success, ""
    np.testing.assert_allclose(np.reshape(result.x, (-1,)), np.reshape(Objective.xmin(n), (-1,)), rtol=1e-7)

# TODO: test when jac is Bool

@pytest.mark.parametrize("shape",[(10,),(10,1),(1,10),(2,4)])
def test_shape_unchanged(shape):

    size = np.prod(shape)
    Objective = mp.Extended_Rosenbrock

    x0 = Objective.startpoint(size).reshape(shape)


    result=LBFGS(Objective.f, x0, jac=Objective.grad, gtol=1e-8)

    assert result.success, ""
    np.testing.assert_allclose(np.reshape(result.x, (-1,)), np.reshape(Objective.xmin(size), (-1,)), rtol=1e-5)

    assert result.x.shape ==shape, "shape of result mismatch shape of startpoint"
    assert result.jac.shape == shape, "shape of result jac mismatch shape of startpoint"

# FIXME: Implement
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