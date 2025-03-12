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
        data = np.random.random(nb_domain_grid_pts)
        assert data.flags["C_CONTIGUOUS"]
    elif order == "F":
        data = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert data.flags["F_CONTIGUOUS"]
    if rank == 0:
        np.save("test_fileload_2d.npy", data)

    comm.barrier()

    yield data
    comm.barrier()
    if rank == 0:
        os.remove("test_fileload_2d.npy")


@pytest.fixture(params=("C", "F"))
def globaldata3d(request, comm):
    order = request.param

    rank = comm.Get_rank()

    nb_domain_grid_pts = (13, 27, 9)
    np.random.seed(2)
    if order == "C":
        data = np.random.random(nb_domain_grid_pts)
        assert data.flags.c_contiguous
    elif order == "F":
        data = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert data.flags.f_contiguous
    if rank == 0:
        np.save("test_3d.npy", data)

    comm.barrier()

    yield data
    comm.barrier()
    if rank == 0:
        os.remove("test_3d.npy")


@pytest.fixture(
    params=[
        ("C", (13, 7)),
        ("F", (13, 7)),
        ("C", (13, 11, 27)),
        ("F", (13, 11, 27)),
        ("C", (15, 7, 9, 3)),
        ("F", (15, 7, 9, 3)),
    ]
)
def datagrid(request, comm):
    order, nb_domain_grid_pts = request.param

    np.random.seed(2)
    if order == "C":
        data = np.random.random(nb_domain_grid_pts)
        assert data.flags.c_contiguous
    elif order == "F":
        data = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert data.flags.f_contiguous

    return data
