import os

import numpy as np
import pytest
from primefac import primefac


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
def globaldata(request, comm):
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


class DistributedData:
    def __init__(self, data, nb_domain_grid_pts, subdomain_locations):
        self.data = data
        self.nb_domain_grid_pts = nb_domain_grid_pts
        self.nb_subdomain_grid_pts = data.shape
        self.subdomain_locations = subdomain_locations

    @property
    def subdomain_slices(self):
        return tuple(
            slice(s, s + n)
            for s, n in zip(self.subdomain_locations, self.nb_subdomain_grid_pts)
        )


def make_2d_slab_x(comm, globaldata):
    """
    Returns the part of globaldata attribute to the present rank in 2D data
    decomposition.

    Parameters
    ----------
    comm : communicator
        The MPI communicator.
    globaldata : numpy.ndarray
        The global data array to be decomposed.

    Returns
    -------
    DistributedData
        The part of the global data assigned to the current rank.
    """
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nb_domain_grid_pts = globaldata.shape

    step = nb_domain_grid_pts[0] // nprocs

    if rank == nprocs - 1:
        subdomain_slices = (slice(rank * step, None), slice(None, None))
        subdomain_locations = [rank * step, 0]
        # nb_subdomain_grid_pts = [nb_domain_grid_pts[0] - rank * step,
        #                          nb_domain_grid_pts[1]]
    else:
        subdomain_slices = (slice(rank * step, (rank + 1) * step), slice(None, None))
        subdomain_locations = [rank * step, 0]
        # nb_subdomain_grid_pts = [step, nb_domain_grid_pts[1]]

    return DistributedData(
        globaldata[subdomain_slices], nb_domain_grid_pts, subdomain_locations
    )


def make_2d_slab_y(comm, globaldata):
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nb_domain_grid_pts = globaldata.shape

    step = nb_domain_grid_pts[1] // nprocs
    if rank == nprocs - 1:
        subdomain_slices = (slice(None, None), slice(rank * step, None))
        subdomain_locations = [0, rank * step]
        # nb_subdomain_grid_pts = [nb_domain_grid_pts[0],
        #                          nb_domain_grid_pts[1] - rank * step]
    else:
        subdomain_slices = (slice(None, None), slice(rank * step, (rank + 1) * step))
        subdomain_locations = [0, rank * step]
        # nb_subdomain_grid_pts = [nb_domain_grid_pts[1], step]

    return DistributedData(
        globaldata[subdomain_slices], nb_domain_grid_pts, subdomain_locations
    )


def suggest_subdivisions(nb_dims, nb_procs):
    facs = list(primefac(nb_procs))
    if len(facs) < nb_dims:
        return facs + [1] * (nb_dims - len(facs))
    return facs[: nb_dims - 1] + [np.prod(facs[nb_dims - 1 :])]


def get_coord(rank, subdivisions):
    coord = []
    for n in subdivisions:
        coord += [rank % n]
        rank = rank // n
    return coord


def subdivide(comm, globaldata):
    """
    Returns the part of the `globaldata` array distributed on a grid.

    Parameters
    ----------
    comm : communicator
        The MPI communicator.
    globaldata : numpy.ndarray
        The global data array to be decomposed.

    Returns
    -------
    DistributedData
        The part of the global data assigned to the current rank.
    """
    nb_domain_grid_pts = globaldata.shape
    subdivisions = suggest_subdivisions(len(nb_domain_grid_pts), comm.Get_size())
    coord = get_coord(comm.Get_rank(), subdivisions)
    nb_subdomain_grid_pts = np.array(globaldata.shape) // subdivisions
    nb_subdomain_grid_pts = tuple(
        n if c < s - 1 else m - n * (s - 1)
        for c, s, n, m in zip(
            coord, subdivisions, nb_subdomain_grid_pts, nb_domain_grid_pts
        )
    )

    subdomain_locations = tuple(n * c for n, c in zip(nb_subdomain_grid_pts, coord))
    subdomain_slices = tuple(
        slice(s, s + n) for s, n in zip(subdomain_locations, nb_subdomain_grid_pts)
    )

    return DistributedData(
        globaldata[subdomain_slices], nb_domain_grid_pts, subdomain_locations
    )
