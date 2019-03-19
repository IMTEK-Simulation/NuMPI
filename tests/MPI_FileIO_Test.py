

import numpy as np
import os

try :
    from mpi4py import MPI
    _withMPI=True

except ImportError:
    print("No MPI")
    _withMPI =False

from MPITools.FileIO.MPIFileIO import save_npy, load_npy,  MPIFileIncompatibleResolutionError, MPIFileViewNPY
import pytest

@pytest.mark.xfail(reason="not implemented", run=False)
def test_FileSave_1D(comm):

    domain_resolution    = 128
    np.random.seed(2)
    globaldata = np.random.random(domain_resolution)

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    step = domain_resolution // nprocs

    if rank == nprocs - 1:
        subdomain_slice = slice(rank * step, None)
        subdomain_location =rank * step
        subdomain_resolution = domain_resolution - rank * step
    else:
        subdomain_slice = slice(rank * step, (rank + 1) * step)
        subdomain_location = rank * step
        subdomain_resolution = step

    localdata = globaldata[subdomain_slice]

    save_npy("test_Filesave_1D.npy",localdata,subdomain_location,domain_resolution,comm)

    loaded_data = np.load("test_Filesave_1D.npy")
    np.testing.assert_array_equal(loaded_data,globaldata)

    comm.barrier()
    if rank == 0:
        os.remove("test_Filesave_1D.npy")


@pytest.fixture(scope="module")
def globaldata(comm):

    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    domain_resolution = (128,128)
    np.random.seed(2)
    globaldata = np.random.random(domain_resolution)

    if rank == 0:
        np.save("test_FileLoad_2D.npy", globaldata)

    comm.barrier()

    yield globaldata

    if rank == 0:
        os.remove("test_FileLoad_2D.npy")


### Helper function:

class DistributedData:
    def __init__(self, data, domain_resolution, subdomain_location):
        self.data = data
        self.domain_resolution = domain_resolution
        self.subdomain_resolution = data.shape
        self.subdomain_location = subdomain_location

    @property
    def subdomain_slice(self):
        return tuple([slice(s, s + n) for s, n in
                      zip(self.subdomain_location, self.subdomain_resolution)])

def make_2d_slab_y(comm, globaldata):

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    domain_resolution = globaldata.shape

    step = domain_resolution[1] // nprocs
    if rank == nprocs - 1:
        subdomain_slice = (slice(None, None), slice(rank * step, None))
        subdomain_location = [0, rank * step]
        subdomain_resolution = [domain_resolution[0], domain_resolution[1] - rank * step]
    else:
        subdomain_slice = (slice(None, None), slice(rank * step, (rank + 1) * step))
        subdomain_location = [0, rank * step]
        subdomain_resolution = [domain_resolution[1], step]

    return DistributedData(globaldata[subdomain_slice], domain_resolution, subdomain_location)

def make_2d_slab_x(comm, globaldata):
    """
    returns the part of globaldata attribute to the present rank in 2D data decomposition
    Parameters
    ----------
    comm : communicator
    globaldata

    Returns
    -------
    DistributedData
    """
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    domain_resolution = globaldata.shape

    step = domain_resolution[0] // nprocs

    if rank == nprocs - 1:
        subdomain_slice = (slice(rank * step, None), slice(None, None))
        subdomain_location = [rank * step, 0]
        subdomain_resolution = [domain_resolution[0] - rank * step, domain_resolution[1]]
    else:
        subdomain_slice = (slice(rank * step, (rank + 1) * step), slice(None, None))
        subdomain_location = [rank * step, 0]
        subdomain_resolution = [step, domain_resolution[1]]

    return DistributedData(globaldata[subdomain_slice], domain_resolution, subdomain_location)

@pytest.mark.parametrize("decompfun",[make_2d_slab_x, make_2d_slab_y])
def test_FileSave_2D(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    save_npy("test_Filesave_2D.npy",distdata.data,distdata.subdomain_location,distdata.domain_resolution, comm)
    loaded_data = np.load("test_Filesave_2D.npy")
    np.testing.assert_array_equal(loaded_data, globaldata)

    comm.barrier()
    if comm.Get_rank() == 0:
        os.remove("test_Filesave_2D.npy")

@pytest.mark.parametrize("decompfun",[make_2d_slab_x, make_2d_slab_y])
def test_FileView_2D(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    #arr = np.load("test_FileLoad_2D.npy")
    #assert arr.shape == self.domain_resolution

    file = MPIFileViewNPY("test_FileLoad_2D.npy", comm=comm)

    assert file.resolution == distdata.domain_resolution
    assert file.dtype == globaldata.dtype

    loaded_data = file.read(subdomain_resolution=distdata.subdomain_resolution,
                            subdomain_location= distdata.subdomain_location)


    np.testing.assert_array_equal(loaded_data, distdata.data)

@pytest.mark.parametrize("decompfun",[make_2d_slab_x, make_2d_slab_y])
def test_FileLoad_2D(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    loaded_data = load_npy("test_FileLoad_2D.npy",
                           subdomain_resolution=distdata.subdomain_resolution,
                           subdomain_location=distdata.subdomain_location,
                           domain_resolution=distdata.domain_resolution,
                           comm=comm)

    np.testing.assert_array_equal(loaded_data, distdata.data)

    with pytest.raises(MPIFileIncompatibleResolutionError):
        load_npy("test_FileLoad_2D.npy",
                 subdomain_resolution=distdata.subdomain_resolution,
                 subdomain_location=distdata.subdomain_location,
                 domain_resolution=tuple([a + 1 for a in distdata.domain_resolution]),
                 comm=comm)