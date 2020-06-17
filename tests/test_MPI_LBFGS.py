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


import pytest

import numpy as np
from NuMPI.Tools import  Reduction
from runtests.mpi import MPITest
from tests.MPI_minimization_problems import MPI_Objective_Interface
import scipy.optimize
import  time
import tests.minimization_problems as mp

from NuMPI.Optimization.MPI_LBFGS_Matrix_H import steepest_descent_wolfe2, LBFGS

def timer(fun, *args, **kwargs):
    start = time.perf_counter()
    res = fun(*args, **kwargs)
    delay = time.perf_counter() - start
    return res, delay

def test_linesearch():
    pass


def test_MPI_Parallel_Interface(comm): # TODO: Duplicate with test_LBFGS_interface ?
    """ Tests the parallel wrapping of objectives
    Test if parallel Version gives the same as the serial Version
    :param comm:
    :return:
    """

    def printMPI(msg):
        for i in range(comm.Get_size()):
            comm.barrier()
            if comm.Get_rank() == i:
                print("Proc {}: {}".format(i, msg))
    n = 10

    par = MPI_Objective_Interface(mp.Extended_Rosenbrock,nb_domain_grid_pts=n,comm=comm)

    printMPI(par.counts)

    ref = mp.Extended_Rosenbrock

    np.testing.assert_array_equal(mp.Extended_Rosenbrock.startpoint(n)[par.subdomain_slices], par.startpoint())
    np.testing.assert_almost_equal(mp.Extended_Rosenbrock.f(mp.Extended_Rosenbrock.startpoint(n)),par.f(par.startpoint()),err_msg="Different Function Value at startpoint")
    np.testing.assert_allclose(mp.Extended_Rosenbrock.grad(mp.Extended_Rosenbrock.startpoint(n))[par.subdomain_slices],
                                   par.grad(par.startpoint()), err_msg="Different Gradient Value at startpoint")


#@pytest.mark.parametrize("PObjective",[mp.Extended_Rosenbrock]) # Objective should support parallelization !
#@pytest.mark.parametrize("n",[10,20])

def test_analytical_min(comm):
    """
    Compares the result with the analyticaly known posistion of the minimum
    :return:
    """

    def printMPI(msg):
        for i in range(comm.Get_size()):
            comm.barrier()
            if comm.Get_rank() == i:
                print("Proc {}: {}".format(i, msg))

    n = 50

    Objective = mp.Extended_Rosenbrock

    PObjective=MPI_Objective_Interface(Objective, nb_domain_grid_pts=n, comm=comm)

    x0 = PObjective.startpoint()

    res = LBFGS(PObjective.f_grad, x0, jac=True, maxcor=5, maxiter=100,
                gtol=1e-12, ftol=0, pnp= Reduction(comm))
    #                        ^ only terminates if gradient condition is satisfied
    assert res.success
    assert res.message=="CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL"
    np.testing.assert_allclose(res.x,PObjective.xmin(),atol=1e-16,rtol = 1e-7)

    assert np.abs(res.fun-Objective.minVal(n))< 1e-7


#def test_ftol(comm):
#    pass

#def test_gtol(comm):
#    pass

#def test_g2tol(comm):
#    pass

#def test_alltol(comm):
#    pass

@pytest.mark.skip(reason="just plotting")
def test_time_complexity(comm):


    maxcor = 5
    Objective = mp.Extended_Rosenbrock
    n= np.array([10,100,1000,1e4,1e5,1e6],dtype = int)
    t = np.zeros(len(n),dtype=float)
    res = [None] * len(n)
    pnp = Reduction(comm)
    for i in range(len(n)):
        PObjective = MPI_Objective_Interface(Objective, nb_domain_grid_pts=n[i], comm=comm)
        x0 = PObjective.startpoint()

        res[i],t[i] = timer(LBFGS,PObjective.f_grad, x0, jac=True, maxcor=maxcor, maxiter=100000, gtol=(1e-5), pnp=pnp)

        assert res[i].success


    if True:
        import matplotlib.pyplot as plt

        fig,ax = plt.subplots()
        ax.plot(n,t/n,'+-', label = "time / DOF")
        ax.plot(n, [t[i] / n[i] / res[i].nit for i in range(len(n))], '+-',label = "time per DOF per iteration")
        ax2 = plt.twinx(ax)
        ax2.plot(n,[res[i].nit for i in range(len(n))],'o',label = "nit")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel("t/n (s)")
        ax.set_xlabel("DOF")
        ax.legend()
        ax2.legend()
        #ax.plot(n,n,c='gray')
        plt.show(block = True)
