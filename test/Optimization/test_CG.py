

from test.Optimization.MPI_minimization_problems import MPI_Quadratic
from NuMPI.Tools import Reduction
import numpy as np

from NuMPI.Optimization.bugnicourt_cg import constrained_conjugate_gradients


def test_bugnicourt_cg(comm):

    n = 128

    obj = MPI_Quadratic(n, pnp=Reduction(comm),)

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        x0=xstart,
        communicator=comm
        )
    assert res.success, res.message
    print(res.nit)

def test_bugnicourt_cg_active_bounds(comm):

    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        x0=xstart,
        communicator=comm
        )
    assert res.success, res.message
    print(res.nit)
    print(np.count_nonzero(res.x==0))



def test_bugnicourt_cg_mean_val(comm):

    n = 128

    obj = MPI_Quadratic(n, pnp=Reduction(comm),)

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        x0=xstart,
        mean_val=1.,
        communicator=comm
        )
    assert res.success, res.message
    print(res.nit)


def test_bugnicourt_cg_mean_val_active_bounds(comm):

    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        x0=xstart,
        mean_val=1.,
        communicator=comm
        )
    assert res.success, res.message
    print(res.nit)