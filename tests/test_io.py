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

from NuMPI.IO.MPIFileIO import save_npy, load_npy,  MPIFileIncompatibleResolutionError, MPIFileViewNPY
from NuMPI import MPI
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
        subdomain_slices = slice(rank * step, None)
        subdomain_location =rank * step
        subdomain_resolution = domain_resolution - rank * step
    else:
        subdomain_slices = slice(rank * step, (rank + 1) * step)
        subdomain_location = rank * step
        subdomain_resolution = step

    localdata = globaldata[subdomain_slices]

    save_npy("test_Filesave_1D.npy",localdata,subdomain_location,domain_resolution,comm)

    loaded_data = np.load("test_Filesave_1D.npy")
    np.testing.assert_array_equal(loaded_data,globaldata)

    comm.barrier()
    if rank == 0:
        os.remove("test_Filesave_1D.npy")

@pytest.mark.parametrize('order', )
@pytest.fixture(scope="module", params=("C", "F"))
def globaldata(request, comm):
    order = request.param

    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    domain_resolution = (128,128)
    np.random.seed(2)
    if order=="C":
        globaldata = np.random.random(domain_resolution)
        assert globaldata.flags["C_CONTIGUOUS"]
    elif order=="F":
        globaldata = np.random.random(domain_resolution[::-1]).transpose()
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
    def __init__(self, data, domain_resolution, subdomain_location):
        self.data = data
        self.domain_resolution = domain_resolution
        self.subdomain_resolution = data.shape
        self.subdomain_location = subdomain_location

    @property
    def subdomain_slices(self):
        return tuple([slice(s, s + n) for s, n in
                      zip(self.subdomain_location, self.subdomain_resolution)])

def make_2d_slab_y(comm, globaldata):

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    domain_resolution = globaldata.shape

    step = domain_resolution[1] // nprocs
    if rank == nprocs - 1:
        subdomain_slices = (slice(None, None), slice(rank * step, None))
        subdomain_location = [0, rank * step]
        subdomain_resolution = [domain_resolution[0], domain_resolution[1] - rank * step]
    else:
        subdomain_slices = (slice(None, None), slice(rank * step, (rank + 1) * step))
        subdomain_location = [0, rank * step]
        subdomain_resolution = [domain_resolution[1], step]

    return DistributedData(globaldata[subdomain_slices], domain_resolution, subdomain_location)

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
        subdomain_slices = (slice(rank * step, None), slice(None, None))
        subdomain_location = [rank * step, 0]
        subdomain_resolution = [domain_resolution[0] - rank * step, domain_resolution[1]]
    else:
        subdomain_slices = (slice(rank * step, (rank + 1) * step), slice(None, None))
        subdomain_location = [rank * step, 0]
        subdomain_resolution = [step, domain_resolution[1]]

    return DistributedData(globaldata[subdomain_slices], domain_resolution, subdomain_location)

@pytest.mark.parametrize("decompfun",[make_2d_slab_x, make_2d_slab_y])
def test_FileSave_2D(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    save_npy("test_Filesave_2D.npy",
             distdata.data,
             distdata.subdomain_location,
             distdata.domain_resolution, comm)
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
                           comm=comm)

    np.testing.assert_array_equal(loaded_data, distdata.data)


@pytest.fixture
def npyfile():
    yield "test_same_numpy.npy"
    try:
        os.remove("test_same_numpy.npy")
    except:
        pass

def test_detect_fortran_order(comm_self):
    # fix some statements on numpy behaviour

    ## Prepare fortran array
    arr = np.array(range(6), dtype=float).reshape(3, 2)
    arr = arr.transpose()

    # States numpy behaviour
    assert arr.shape == (2, 3)
    assert arr.flags["C_CONTIGUOUS"]==False
    assert arr.flags["F_CONTIGUOUS"]==True

    np.save("test.npy", arr)

    #asserts the loaded array is exactly the same
    loaded = np.load("test.npy")
    assert loaded.shape == (2, 3)
    assert loaded.flags["C_CONTIGUOUS"] == False
    assert loaded.flags["F_CONTIGUOUS"] == True

def test_load_same_numpy_load(comm_self, npyfile):
    data = np.random.random(size=(2, 3))
    np.save(npyfile, data)
    loaded_data = load_npy(npyfile, comm=comm_self)
    np.testing.assert_allclose(loaded_data, data)

#@pytest.mark.xfail # see #15
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

@pytest.mark.xfail #see #15
def test_same_numpy_save_transposed(comm_self, npyfile):
    data = np.random.random(size=(2, 3)).T
    save_npy(npyfile, data, comm=comm_self)
    loaded_data = np.load(npyfile)
    np.testing.assert_allclose(loaded_data, data)
    assert np.isfortran(data) == np.isfortran(loaded_data)


