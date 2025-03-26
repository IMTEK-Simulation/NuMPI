import numpy as np
import pytest

from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert

try:
    import scipy.optimize

    _scipy_present = True
except ModuleNotFoundError:
    _scipy_present = False

from test.Optimization.MPIMinimizationProblems import MPI_Quadratic

from NuMPI.Optimization.CCGWithoutRestart import \
    constrained_conjugate_gradients
from NuMPI.Tools import Reduction


def test_bugnicourt_cg(comm):
    n = 128

    obj = MPI_Quadratic(n, pnp=Reduction(comm), )

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)


@pytest.mark.skipif(not _scipy_present, reason='scipy not present')
def test_bugnicourt_cg_arbitrary_bounds(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), )

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)
    bounds = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        communicator=comm,
        bounds=bounds,
        gtol=1e-8
    )
    parallel_assert(comm, res.success, res.message)

    parallel_assert(comm, (res.x >= bounds).all())
    print(res.nit)

    # TODO: we are not checking yet that the result is the same in parallel.
    if comm.size == 1:
        bnds = tuple([(b, None) for b in bounds])

        res_ref = scipy.optimize.minimize(
            obj.f_grad,
            args=(2,),
            x0=xstart,
            bounds=bnds,
            method="L-BFGS-B",
            jac=True,
            options=dict(gtol=1e-8, ftol=0))
        parallel_assert(comm, res_ref.success, res_ref.message)

        assert_all_allclose(comm, res.x, res_ref.x, atol=1e-6)


def test_bugnicourt_cg_active_bounds(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)
    print(np.count_nonzero(res.x == 0))


def test_bugnicourt_cg_mean_val(comm):
    n = 128

    obj = MPI_Quadratic(n, pnp=Reduction(comm), )

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        mean_val=1.,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)


def test_bugnicourt_cg_mean_val_active_bounds(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        mean_val=1.,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)
