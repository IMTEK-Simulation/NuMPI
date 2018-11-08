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

    for i in range(12):
        x= Objective.bounds[0] + (Objective.bounds[1] - Objective.bounds[0]) * np.random.random(n)
        x.shape = (-1,1)
        u=np.random.normal(size=n) # Direction
        u /= np.linalg.norm(u,2) # normalize
        u.shape = (-1,1)

        # TODO: Don't know which Tolerance to associate with which eps
        eps = 1e-5
        print((Objective.f(x + u * eps) - Objective.f(x)) / eps)
        der_numerical  = (Objective.f(x + u * eps) - Objective.f(x)) / eps
        der_analytical = np.asscalar(Objective.grad(x).T@u)
        assert abs(der_numerical- der_analytical)/der_analytical < 1e-3, "(der_numerical- der_analytical)/der_analytical = {}".format(abs(der_numerical- der_analytical)/der_analytical)

@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock,mp.Trigonometric])
@pytest.mark.parametrize("n",[2,10,20])
def test_Gradient(Objective,n) :
    """
    Asserts the numerical Gradient converges with order of the step to the analyitical one.

    This uses check_gradient from scipy

    :param Objective: which function to minimize
    :param n: number of dimensions
    :return:
    """

    for i in range(12):
        x= Objective.bounds[0] + (Objective.bounds[1] - Objective.bounds[0]) * np.random.random(n)
        #x.shape = (-1,1)

        epsilons = np.array([1e-3,1e-5,1e-6])
        errors = np.array([scipy.optimize.check_grad(Objective.f,Objective.grad,x,epsilon = eps) for eps in epsilons ]) # should be O(epsilon)
        errorratios = errors / epsilons # should be O(1)
        errorratios /= errorratios[0]
        #print(errorratios)
        assert np.prod(errorratios < 10), "errorratios = {}".format(errorratios)

