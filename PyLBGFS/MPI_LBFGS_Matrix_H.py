
import numpy as np
import scipy.optimize

from Tools.ParallelNumpy import ParallelNumpy
from mpi4py import MPI

def donothing(*args,**kwargs):
    pass

def steepest_descent_wolfe2(x0,f,fprime, pnp = ParallelNumpy,**kwargs):
    """
    For first Iteration there is no history. We make a steepest descent satisfying strong Wolfe Condition
    :return:
    """

    # x_old.shape=(-1,1)
    grad0 = fprime(x0)

    # line search
    alpha, phi, phi0, derphi = scipy.optimize.linesearch.scalar_search_wolfe2(
        lambda alpha: f(x0 - grad0 * alpha),
        lambda alpha: pnp.dot(fprime(x0 - grad0 * alpha).T, -grad0),**kwargs)
    assert derphi is not None, "Line Search in first steepest descent failed"
    x = x0 - grad0 * alpha

    return x, fprime(x) , x0, grad0

