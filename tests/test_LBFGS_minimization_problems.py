#
# Copyright 2018 Antoine Sanner
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
serial test
"""

import numpy as np
import scipy.optimize
from NuMPI.Optimization.LBFGS_Matrix_H import LBFGS
import tests.minimization_problems as mp

import pytest
@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock]) # Only these where Solution is analytically known
@pytest.mark.parametrize("n",[10,20])
def test_analytical_min(Objective,n):
    """
    Compares the result with the analyticaly known posistion of the minimum
    """

    x0 = Objective.startpoint(n)

    res = LBFGS(Objective.f, x0, jac=Objective.grad, maxcor=5, gtol=1e-6, maxiter=1000)

    np.testing.assert_allclose(np.reshape(res.x, (-1,)), np.reshape(Objective.xmin(n),(-1,)), rtol=1e-6)

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