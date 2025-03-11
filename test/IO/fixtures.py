import os

import numpy as np
import pytest


@pytest.fixture
def npyfile():
    """
    defines a filename and makes a cleanup once the test was executed
    Yields
    ------
    filename

    """
    yield "test_same_numpy.npy"
    try:
        os.remove("test_same_numpy.npy")
    except FileNotFoundError:
        pass


@pytest.fixture(params=("C", "F"))
def globaldata2d(request, comm):
    order = request.param

    rank = comm.Get_rank()

    nb_domain_grid_pts = (128, 128)
    np.random.seed(2)
    if order == "C":
        globaldata = np.random.random(nb_domain_grid_pts)
        assert globaldata.flags["C_CONTIGUOUS"]
    elif order == "F":
        globaldata = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert globaldata.flags["F_CONTIGUOUS"]
    if rank == 0:
        np.save("test_fileload_2d.npy", globaldata)

    comm.barrier()

    yield globaldata
    comm.barrier()
    if rank == 0:
        os.remove("test_fileload_2d.npy")


@pytest.fixture(params=("C", "F"))
def globaldata3d(request, comm):
    order = request.param

    rank = comm.Get_rank()

    nb_domain_grid_pts = (32, 17, 23)
    np.random.seed(2)
    if order == "C":
        globaldata = np.random.random(nb_domain_grid_pts)
        assert globaldata.flags["C_CONTIGUOUS"]
    elif order == "F":
        globaldata = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert globaldata.flags["F_CONTIGUOUS"]
    if rank == 0:
        np.save("test_3d.npy", globaldata)

    comm.barrier()

    yield globaldata
    comm.barrier()
    if rank == 0:
        os.remove("test_3d.npy")
