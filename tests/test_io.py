#
# Copyright 2019 Lars Pastewka
#           2018-2019 Antoine Sanner
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



import numpy as np
import os
import warnings

from NuMPI.IO.MPIFileIO import save_npy, load_npy,  MPIFileIncompatibleResolutionError, MPIFileViewNPY, MPIFileTypeError
from NuMPI import MPI
import NuMPI
import pytest

testdir = os.path.dirname(os.path.realpath(__file__))

def test_FileSave_1D(comm):
    nb_domain_grid_pts = 128
    np.random.seed(2)
    globaldata = np.random.random(nb_domain_grid_pts)

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    step = nb_domain_grid_pts // nprocs

    if rank == nprocs - 1:
        subdomain_slices = slice(rank * step, None)
        subdomain_locations =rank * step
        nb_subdomain_grid_pts = nb_domain_grid_pts - rank * step
    else:
        subdomain_slices = slice(rank * step, (rank + 1) * step)
        subdomain_locations = rank * step
        nb_subdomain_grid_pts = step

    localdata = globaldata[subdomain_slices]

    save_npy("test_Filesave_1D.npy",
             localdata,
             subdomain_locations,
             nb_domain_grid_pts,
             comm)
    comm.barrier() # The MPI_File reading and closing doesn't have to finish together
    loaded_data = np.load("test_Filesave_1D.npy")
    np.testing.assert_array_equal(loaded_data,globaldata)

    comm.barrier()
    if rank == 0:
        os.remove("test_Filesave_1D.npy")

@pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="test is only serial")
def test_FileSave_1D_list():
    nb_domain_grid_pts = 8
    np.random.seed(2) 
    globaldata = np.random.random(nb_domain_grid_pts).tolist()

    save_npy("test_Filesave_1D_list.npy",globaldata)

    loaded_data = np.load("test_Filesave_1D_list.npy")
    np.testing.assert_array_equal(loaded_data,globaldata)

    os.remove("test_Filesave_1D_list.npy")

@pytest.fixture(scope="module", params=("C", "F"))
def globaldata(request, comm):
    order = request.param

    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    nb_domain_grid_pts = (128,128)
    np.random.seed(2)
    if order=="C":
        globaldata = np.random.random(nb_domain_grid_pts)
        assert globaldata.flags["C_CONTIGUOUS"]
    elif order=="F":
        globaldata = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert globaldata.flags["F_CONTIGUOUS"]
    if rank == 0:
        np.save("test_FileLoad_2D.npy", globaldata)

    comm.barrier()

    yield globaldata
    comm.barrier()
    if rank == 0:
        os.remove("test_FileLoad_2D.npy")


### Helper function:

class DistributedData:
    def __init__(self, data, nb_domain_grid_pts, subdomain_locations):
        self.data = data
        self.nb_domain_grid_pts = nb_domain_grid_pts
        self.nb_subdomain_grid_pts = data.shape
        self.subdomain_locations = subdomain_locations

    @property
    def subdomain_slices(self):
        return tuple([slice(s, s + n) for s, n in
                      zip(self.subdomain_locations, self.nb_subdomain_grid_pts)])

def make_2d_slab_y(comm, globaldata):

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nb_domain_grid_pts = globaldata.shape

    step = nb_domain_grid_pts[1] // nprocs
    if rank == nprocs - 1:
        subdomain_slices = (slice(None, None), slice(rank * step, None))
        subdomain_locations = [0, rank * step]
        nb_subdomain_grid_pts = [nb_domain_grid_pts[0], nb_domain_grid_pts[1] - rank * step]
    else:
        subdomain_slices = (slice(None, None), slice(rank * step, (rank + 1) * step))
        subdomain_locations = [0, rank * step]
        nb_subdomain_grid_pts = [nb_domain_grid_pts[1], step]

    return DistributedData(globaldata[subdomain_slices], nb_domain_grid_pts, subdomain_locations)

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

    nb_domain_grid_pts = globaldata.shape

    step = nb_domain_grid_pts[0] // nprocs

    if rank == nprocs - 1:
        subdomain_slices = (slice(rank * step, None), slice(None, None))
        subdomain_locations = [rank * step, 0]
        nb_subdomain_grid_pts = [nb_domain_grid_pts[0] - rank * step, nb_domain_grid_pts[1]]
    else:
        subdomain_slices = (slice(rank * step, (rank + 1) * step), slice(None, None))
        subdomain_locations = [rank * step, 0]
        nb_subdomain_grid_pts = [step, nb_domain_grid_pts[1]]

    return DistributedData(globaldata[subdomain_slices], nb_domain_grid_pts, subdomain_locations)

@pytest.mark.parametrize("decompfun",[make_2d_slab_x, make_2d_slab_y])
def test_FileSave_2D(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    save_npy("test_Filesave_2D.npy",
             distdata.data,
             distdata.subdomain_locations,
             distdata.nb_domain_grid_pts, comm)
    comm.barrier()
    if comm.Get_rank() == 0:
        loaded_data = np.load("test_Filesave_2D.npy")
        np.testing.assert_array_equal(loaded_data, globaldata)

        os.remove("test_Filesave_2D.npy")
    comm.barrier()

@pytest.mark.parametrize("decompfun",[make_2d_slab_x, make_2d_slab_y])
def test_FileView_2D(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    #arr = np.load("test_FileLoad_2D.npy")
    #assert arr.shape == self.nb_domain_grid_pts

    file = MPIFileViewNPY("test_FileLoad_2D.npy", comm=comm)

    assert file.nb_grid_pts == distdata.nb_domain_grid_pts
    assert file.dtype == globaldata.dtype

    loaded_data = file.read(nb_subdomain_grid_pts=distdata.nb_subdomain_grid_pts,
                            subdomain_locations= distdata.subdomain_locations)


    np.testing.assert_array_equal(loaded_data, distdata.data)


@pytest.mark.parametrize("decompfun",[make_2d_slab_x, make_2d_slab_y])
def test_FileLoad_2D(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    loaded_data = load_npy("test_FileLoad_2D.npy",
                           nb_subdomain_grid_pts=distdata.nb_subdomain_grid_pts,
                           subdomain_locations=distdata.subdomain_locations,
                           comm=comm)

    np.testing.assert_array_equal(loaded_data, distdata.data)


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

@pytest.mark.skip(reason="just some statements on numpy behaviour")
def test_detect_fortran_order(comm_self):
    # fix some statements on numpy behaviour

    # Prepare fortran array
    arr = np.array(range(6), dtype=float).reshape(3, 2)
    arr = arr.transpose()

    # States numpy behaviour
    assert arr.shape == (2, 3)
    assert arr.flags["C_CONTIGUOUS"] is False
    assert arr.flags["F_CONTIGUOUS"] is True

    np.save("test.npy", arr)

    # asserts the loaded array is exactly the same
    loaded = np.load("test.npy")
    assert loaded.shape == (2, 3)
    assert loaded.flags["C_CONTIGUOUS"] is False
    assert loaded.flags["F_CONTIGUOUS"] is True


def test_load_same_numpy_load(comm_self, npyfile):
    data = np.random.random(size=(2, 3))
    np.save(npyfile, data)
    loaded_data = load_npy(npyfile, comm=comm_self)
    np.testing.assert_allclose(loaded_data, data)


def test_same_numpy_load_transposed(comm_self, npyfile):
    data = np.random.random(size=(2, 3)).T
    np.save(npyfile, data)
    loaded_data = load_npy(npyfile, comm=comm_self)
    np.testing.assert_allclose(loaded_data, data)


def test_load_same_numpy_save(comm_self,npyfile):
    data = np.random.random(size=(2, 3))
    save_npy(npyfile, data, comm=comm_self)
    loaded_data = np.load(npyfile)
    np.testing.assert_allclose(loaded_data, data)
    assert np.isfortran(data) == np.isfortran(loaded_data)


def test_same_numpy_save_transposed(comm_self, npyfile):
    data = np.random.random(size=(2, 3)).T
    save_npy(npyfile, data, comm=comm_self)
    loaded_data = np.load(npyfile)
    np.testing.assert_allclose(loaded_data, data)
    assert np.isfortran(data) == np.isfortran(loaded_data)


#@pytest.mark.filterwarnings("error: ResourceWarnings")
# unfortunately the ResourceWarnings are never raised even when the files are not closed
def test_raises_and_no_resourcewarnings(comm_self):
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter(
            "always")  # deactivate hiding of ResourceWarnings

        with pytest.raises(MPIFileTypeError):
            load_npy(os.path.join(testdir, "wrongnpyfile.npy"), comm=comm_self)

        # assert no warning is a ResourceWarning
        for wi in w:
            assert not issubclass(wi.category, ResourceWarning)

def test_corrupted_file(comm_self):
    """
    tests that the reader behaves decently when trying to open a file having
    the wrong format see issue #23
    """
    # create test corrupted file

    with open("corrupted.dummy", "w") as f:
        f.write("dfgdfghlkjhgiuhdfg")

    with pytest.raises(MPIFileTypeError):
        MPIFileViewNPY("corrupted.dummy", comm=comm_self)

@pytest.mark.skipif(NuMPI._has_mpi4py, reason="filestreams are not supported when "
"                                         NuMPI is using with mpi4py")
def test_filestream(comm_self, npyfile):
    data = np.random.normal(size=(4,6))

    np.save(npyfile, data)
    with open(npyfile, mode="r") as f:
        read_data = load_npy(f, comm=comm_self)
        np.testing.assert_allclose(read_data, data)
    with open(npyfile, mode="rb") as f:
        read_data = load_npy(f, comm=comm_self)
        np.testing.assert_allclose(read_data, data)