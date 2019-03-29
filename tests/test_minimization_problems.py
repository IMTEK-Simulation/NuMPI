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


import tests.minimization_problems as mp
import numpy as np
import scipy.optimize

import pytest
@pytest.mark.parametrize("Objective",[mp.Trigonometric,mp.Extended_Rosenbrock])
@pytest.mark.parametrize("n",[2,4,10])
def test_directional_derivative(Objective,n) :
    """
    Asserts the nmerical and the Gradient provided by Objective are close
    :param Objective: which function to minimize
    :param n: number of dimensions
    :return:
    """
    _verbose = False
    if _verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

    for i in range(12):
        x= Objective.bounds[0] + (Objective.bounds[1] - Objective.bounds[0]) * np.random.random(n)
        x.shape = (-1,1)
        u=np.random.normal(size=n) # Direction
        u /= np.linalg.norm(u,2) # normalize
        u.shape = (-1,1)

        # TODO: Don't know which Tolerance to associate with which eps
        epsilons = np.array([1e-3,1e-5,1e-6])
        #der_numerical = np.zeros_like(epsilons)
        #der_analytical = np.zeros_like(epsilons)
        errorratios = np.zeros_like(epsilons)
        for eps, i in zip(epsilons, range(len(epsilons))):
            #print((Objective.f(x + u * eps) - Objective.f(x)) / eps)
            der_numerical  = (Objective.f(x + u * eps) - Objective.f(x)) / eps
            der_analytical = (Objective.grad(x).T@u).item()

            errorratios[i] =  np.abs(der_numerical - der_analytical) /eps
        errorratios /= errorratios[0]
        if _verbose:
            ax.plot(epsilons,errorratios)

        assert (errorratios < 10).all(), "error_ratios".format(abs(der_numerical- der_analytical)/der_analytical)
    if _verbose:
        plt.show(block=True)
@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock,mp.Trigonometric])
@pytest.mark.parametrize("n",[2,4,10,20])
def test_Gradient(Objective,n) :
    """
    Asserts the numerical Gradient converges with order of the step to the analyitical one.

    This uses check_gradient from scipy

    :param Objective: which function to minimize
    :param n: number of dimensions
    :return:
    """
    _verbose = False
    if _verbose:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

    for i in range(12): # reproduce to have a bit of statistics
        x= Objective.bounds[0] + (Objective.bounds[1] - Objective.bounds[0]) * np.random.random(n)
        #x.shape = (-1,1)

        epsilons = np.array([1e-3,1e-5,1e-6])
        errors = np.array([scipy.optimize.check_grad(Objective.f,Objective.grad,x,epsilon = eps) for eps in epsilons ]) # should be O(epsilon)
        errorratios = errors / epsilons # should be O(1)
        errorratios /= errorratios[0]
        if _verbose:
            ax.plot(epsilons,errorratios)


        #print(errorratios)
        assert np.prod(errorratios < 10), "errorratios = {}".format(errorratios)
    if _verbose:
        plt.show(block=True)
