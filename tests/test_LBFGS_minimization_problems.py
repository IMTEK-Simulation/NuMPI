import numpy as np
import scipy.optimize
from PyLBGFS.LBFGS_Matrix_H import LBFGS
import tests.minimization_problems as mp

import pytest
@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock]) # Only these where Solution is analytically known
@pytest.mark.parametrize("n",[10,20])
def test_analytical_min(Objective,n):
    """
    Compares the result with the analyticaly known posistion of the minimum
    :return:
    """

    x0 = Objective.startpoint(n)

    res = LBFGS(Objective.f, x0, jac=Objective.grad, maxcor=5, gtol=1e-6, maxiter=1000)

    #np.testing.assert_almost_equal( res.x,Objective.xmin(n))
    np.testing.assert_allclose(np.reshape(res.x, (-1,)), np.reshape(Objective.xmin(n),(-1,)), rtol=1e-7)

@pytest.mark.parametrize("Objective",[mp.Trigonometric])
@pytest.mark.parametrize("n",[10,30])
def test_compare_scipy(Objective,n):
    x0 = Objective.startpoint(n)

    resLBGFS = LBFGS(Objective.f, x0, jac=Objective.grad, maxcor=5, gtol=1e-8, maxiter=1000)
    assert resLBGFS.success
    #Patch
    #resScipy = scipy.optimize.minimize(lambda x: Objective.f(np.reshape(x,(-1,1))),np.reshape(x0,(-1,)), jac = lambda x : np.reshape(Objective.grad(np.reshape(x,(-1,1))),(-1,)),method='L-BFGS-B',tol = 1e-4)
    resScipy = scipy.optimize.minimize(Objective.f, np.reshape(x0, (-1,)),
                                      jac= Objective.grad,method="L-BFGS-B",
                                      options=dict(gtol = 1e-8))
    assert resScipy.success

    np.testing.assert_allclose( np.reshape(resLBGFS.x,(-1,)),resScipy.x, rtol= 1e-2, atol = 1e-5)  #TODO: The location of the minimumseems o be hard to find
    assert np.abs(Objective.f(resScipy.x) - Objective.f(resLBGFS.x)) < 1e-8