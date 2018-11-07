from helpers import minimization_problems as mp
import numpy as np

import pytest
@pytest.mark.parametrize("Objective",[mp.Extended_Rosenbrock,mp.Trigonometric])
@pytest.mark.parametrize("n",[2,4,10])
def test_Gradient(Objective,n) :
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
        der_numerical  = np.asscalar((Objective.f(x + u * eps) - Objective.f(x)) / eps)
        der_analytical = np.asscalar(Objective.grad(x).T@u)
        assert abs(der_numerical- der_analytical)/der_analytical < 1e-3, "(der_numerical- der_analytical)/der_analytical = {}".format(abs(der_numerical- der_analytical)/der_analytical)
