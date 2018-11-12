import pytest

import numpy as np
from mpi4py import MPI
from Tools.ParallelNumpy import  ParallelNumpy
from runtests.mpi import MPITest
from tests.MPI_minimization_problems import MPI_Objective_Interface
import scipy.optimize

import tests.minimization_problems as mp

from PyLBGFS.MPI_LBFGS_Matrix_H import steepest_descent_wolfe2


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
#@MPITest([1, 2, 3, 4])
@pytest.mark.xfail
def test_analytical_min(comm):
    """
    Compares the result with the analyticaly known posistion of the minimum
    :return:
    """
    #comm= MPI.COMM_WORLD
    raise NotImplementedError
    n = 10

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    step = n // nprocs

    if rank == nprocs - 1:
        sl = slice(rank * step, None)
    else:
        sl = slice(rank * step, (rank + 1) * step)

    x0 = PObjective.startpoint(n)[sl]


    res = LBFGS(PObjective.f, PObjective.grad, x0, m=5, MAXITER=100, grad2tol=1e-11,comm = comm)


    np.testing.assert_allclose(res.x,PObjective.xmin(n)[sl])

    recvbuf = None
    if rank == 0:
        recvbuf = np.empty(1,dtype=x0.dtype)
    MPI.COMM_WORLD.Gather(a, recvbuf, root=0)

    assert np.abs(res.fun-PObjective.minVal(n))< 1e-7

    np.testing.assert_almost_equal( res.x,PObjective.xmin(n))