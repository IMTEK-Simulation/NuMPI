import unittest
import numpy as np
import scipy.optimize
from LBFGS_Matrix_H import LBFGS
from helpers import minimization_problems as mp

import pytest
@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock]) # Only these where Solution is analytically known
@pytest.mark.parametrize("n",[10,20])
def test_analytical_min(Objective,n):
    """
    Compares the result with the analyticaly known posistion of the minimum
    :return:
    """

    x0 = Objective.startpoint(n)

    res = LBFGS(Objective.f, Objective.grad, x0, m=5, MAXITER=100, grad2tol=1e-11)

    np.testing.assert_almost_equal( res.x,Objective.xmin(n))

@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock,mp.Trigonometric])
@pytest.mark.parametrize("n",[10,30])
def test_compare_scipy(Objective,n):
    x0 = Objective.startpoint(n)

    resLBGFS = LBFGS(Objective.f, Objective.grad, x0, m=5, MAXITER=100, grad2tol=1e-11)

    print(Objective.grad(x0))

    #Patch
    #resScipy = scipy.optimize.minimize(lambda x: Objective.f(np.reshape(x,(-1,1))),np.reshape(x0,(-1,)), jac = lambda x : np.reshape(Objective.grad(np.reshape(x,(-1,1))),(-1,)),method='L-BFGS-B',tol = 1e-4)
    resScipy = scipy.optimize.minimize(Objective.f, np.reshape(x0, (-1,)),
                                      jac= Objective.grad,
                                      tol=1e-6)
    assert resScipy.success

    np.testing.assert_allclose( np.reshape(resLBGFS.x,(-1,)),resScipy.x, rtol= 1e-3)