import pytest

import numpy as np
from MPITools.Tools import  ParallelNumpy
from runtests.mpi import MPITest
from tests.MPI_minimization_problems import MPI_Objective_Interface
import scipy.optimize
import  time
import tests.minimization_problems as mp

from MPITools.Optimization.MPI_LBFGS_Matrix_H import steepest_descent_wolfe2, LBFGS

def timer(fun, *args, **kwargs):
    start = time.perf_counter()
    res = fun(*args, **kwargs)
    delay = time.perf_counter() - start
    return res, delay

def test_linesearch():
    pass

@MPITest([1, 2, 3, 4])
def test_MPI_Parallel_Interface(comm):
    """
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

    par = MPI_Objective_Interface(mp.Extended_Rosenbrock,domain_resolution=n,comm=comm)

    printMPI(par.counts)

    ref = mp.Extended_Rosenbrock

    np.testing.assert_array_equal(mp.Extended_Rosenbrock.startpoint(n)[par.subdomain_slice], par.startpoint())
    np.testing.assert_almost_equal(mp.Extended_Rosenbrock.f(mp.Extended_Rosenbrock.startpoint(n)),par.f(par.startpoint()),err_msg="Different Function Value at startpoint")
    np.testing.assert_allclose(mp.Extended_Rosenbrock.grad(mp.Extended_Rosenbrock.startpoint(n))[par.subdomain_slice],
                                   par.grad(par.startpoint()), err_msg="Different Gradient Value at startpoint")

@MPITest([1,2,3,4])
def test_MPI_steepest_descent_wolfe2(comm):

    n = 24

    par = MPI_Objective_Interface(mp.Extended_Rosenbrock, domain_resolution=n, comm=comm)
    ref = mp.Extended_Rosenbrock

    start = mp.Extended_Rosenbrock.startpoint(n).reshape(-1)
    direction = - mp.Extended_Rosenbrock.grad(start).reshape(-1)

    alpha =scipy.optimize.line_search(mp.Extended_Rosenbrock.f,mp.Extended_Rosenbrock.grad,start,direction,c1=1e-4,c2=0.9)[0]

    assert alpha is not None

    x_ref = start + direction * alpha
    #steepest_descent_wolfe2

    x= steepest_descent_wolfe2(par.startpoint(),par.f,par.grad,pnp=ParallelNumpy(comm),c1=1e-4,c2=0.9)[0]

    np.testing.assert_allclose(x.reshape(-1), x_ref[par.subdomain_slice])



#@pytest.mark.parametrize("PObjective",[mp.Extended_Rosenbrock]) # Objective should support parallelization !
#@pytest.mark.parametrize("n",[10,20])
@MPITest([1, 2,3,4,8,10,11])
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

    PObjective=MPI_Objective_Interface(Objective, domain_resolution=n, comm=comm)

    x0 = PObjective.startpoint()

    res = LBFGS(PObjective.f, x0,jac=PObjective.grad, maxcor=5, maxiter=100, gtol=1e-12,pnp= ParallelNumpy(comm))

    np.testing.assert_allclose(res.x,PObjective.xmin(),atol=1e-16,rtol = 1e-5)

    assert np.abs(res.fun-Objective.minVal(n))< 1e-7

@MPITest([1, 2,3,4,8,10,11])
def test_ftol(comm):
    pass
@MPITest([1, 2,3,4,8,10,11])
def test_gtol(comm):
    pass
@MPITest([1, 2,3,4,8,10,11])
def test_g2tol(comm):
    pass
@MPITest([1, 2,3,4,8,10,11])
def test_alltol(comm):
    pass

@pytest.mark.skip(reason="just plotting")
def test_time_complexity(comm):


    maxcor = 5
    Objective = mp.Extended_Rosenbrock
    n= np.array([10,100,1000,1e4,1e5,1e6],dtype = int)
    t = np.zeros(len(n),dtype=float)
    res = [None] * len(n)
    pnp = ParallelNumpy(comm)
    for i in range(len(n)):
        PObjective = MPI_Objective_Interface(Objective, domain_resolution=n[i], comm=comm)
        x0 = PObjective.startpoint()

        res[i],t[i] = timer(LBFGS,PObjective.f, x0, jac=PObjective.grad, maxcor=maxcor, maxiter=100000, gtol=(1e-5), pnp=pnp)

        assert res[i].success


    if False:
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
