#
# Copyright 2018, 2020 Antoine Sanner
#           2020 k.o.haase@googlemail.com
#           2019 Lars Pastewka
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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

from NuMPI.IO.MPIFileIO import (
    save_npy,
    load_npy,
    MPIFileViewNPY,
    MPIFileTypeError,
    make_mpi_file_view,
)
from NuMPI import MPI
import NuMPI
import pytest

from .fixtures import (
    make_2d_slab_x,
    make_2d_slab_y,
    subdivide,
    globaldata,
    globaldata3d,
    npyfile,
)

testdir = os.path.dirname(os.path.realpath(__file__))


def test_filesave_1D(comm):
    nb_domain_grid_pts = 128
    np.random.seed(2)
    globaldata = np.random.random(nb_domain_grid_pts)

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    step = nb_domain_grid_pts // nprocs

    if rank == nprocs - 1:
        subdomain_slices = slice(rank * step, None)
        subdomain_locations = rank * step
        # nb_subdomain_grid_pts = nb_domain_grid_pts - rank * step
    else:
        subdomain_slices = slice(rank * step, (rank + 1) * step)
        subdomain_locations = rank * step
        # nb_subdomain_grid_pts = step

    localdata = globaldata[subdomain_slices]

    save_npy(
        "test_Filesave_1D.npy",
        localdata,
        (subdomain_locations,),
        (nb_domain_grid_pts,),
        comm,
    )
    comm.barrier()  # The MPI_File reading and closing doesn't have to
    # finish together
    loaded_data = np.load("test_Filesave_1D.npy")
    np.testing.assert_array_equal(loaded_data, globaldata)

    comm.barrier()
    if rank == 0:
        os.remove("test_Filesave_1D.npy")


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1, reason="test is only serial")
def test_filesave_1D_list():
    nb_domain_grid_pts = 8
    np.random.seed(2)
    globaldata = np.random.random(nb_domain_grid_pts).tolist()

    save_npy("test_Filesave_1D_list.npy", globaldata, nb_grid_pts=(nb_domain_grid_pts,))

    loaded_data = np.load("test_Filesave_1D_list.npy")
    np.testing.assert_array_equal(loaded_data, globaldata)

    os.remove("test_Filesave_1D_list.npy")


@pytest.mark.parametrize("decompfun", [make_2d_slab_x, make_2d_slab_y])
def test_filesave_2d(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    save_npy(
        "test_filesave_2d.npy",
        distdata.data,
        distdata.subdomain_locations,
        distdata.nb_domain_grid_pts,
        comm,
    )
    comm.barrier()
    if comm.Get_rank() == 0:
        loaded_data = np.load("test_filesave_2d.npy")
        np.testing.assert_array_equal(loaded_data, globaldata)

        os.remove("test_filesave_2d.npy")
    comm.barrier()


@pytest.mark.parametrize("decompfun", [make_2d_slab_x, make_2d_slab_y])
def test_fileview_2d(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    # arr = np.load("test_fileload_2d.npy")
    # assert arr.shape == self.nb_domain_grid_pts

    file = MPIFileViewNPY("test_fileload_2d.npy", comm=comm)

    assert file.nb_grid_pts == distdata.nb_domain_grid_pts
    assert file.dtype == globaldata.dtype

    loaded_data = file.read(
        nb_subdomain_grid_pts=distdata.nb_subdomain_grid_pts,
        subdomain_locations=distdata.subdomain_locations,
    )

    np.testing.assert_array_equal(loaded_data, distdata.data)


@pytest.mark.parametrize("decompfun", [make_2d_slab_x, make_2d_slab_y])
def test_fileload_2d(decompfun, comm, globaldata):
    distdata = decompfun(comm, globaldata)

    loaded_data = load_npy(
        "test_fileload_2d.npy",
        nb_subdomain_grid_pts=distdata.nb_subdomain_grid_pts,
        subdomain_locations=distdata.subdomain_locations,
        comm=comm,
    )

    np.testing.assert_array_equal(loaded_data, distdata.data)


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


def test_load_same_numpy_save(comm_self, npyfile):
    data = np.random.random(size=(2, 3))
    save_npy(npyfile, data, nb_grid_pts=data.shape, comm=comm_self)
    loaded_data = np.load(npyfile)
    np.testing.assert_allclose(loaded_data, data)
    assert np.isfortran(data) == np.isfortran(loaded_data)


def test_same_numpy_save_transposed(comm_self, npyfile):
    data = np.random.random(size=(2, 3)).T
    save_npy(npyfile, data, nb_grid_pts=data.shape, comm=comm_self)
    loaded_data = np.load(npyfile)
    np.testing.assert_allclose(loaded_data, data)
    assert np.isfortran(data) == np.isfortran(loaded_data)


# @pytest.mark.filterwarnings("error: ResourceWarnings")
# unfortunately the ResourceWarnings are never raised even when the files
# are not closed
def test_raises_and_no_resourcewarnings(comm_self):
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")  # deactivate hiding of ResourceWarnings

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


@pytest.mark.skipif(
    NuMPI._has_mpi4py,
    reason="filestreams are not supported when "
    "                                         NuMPI "
    "is using with mpi4py",
)
def test_filestream(comm_self, npyfile):
    data = np.random.normal(size=(4, 6))

    np.save(npyfile, data)
    with open(npyfile, mode="r") as f:
        read_data = load_npy(f, comm=comm_self)
        np.testing.assert_allclose(read_data, data)
    with open(npyfile, mode="rb") as f:
        read_data = load_npy(f, comm=comm_self)
        np.testing.assert_allclose(read_data, data)


@pytest.mark.skipif(
    NuMPI._has_mpi4py,
    reason="filestreams are not supported when NuMPI " "is using with mpi4py",
)
@pytest.mark.parametrize("mode", ["r"] if NuMPI._has_mpi4py else ["r", "rb"])
def test_make_mpi_file_view(comm_self, npyfile, mode):
    data = np.random.normal(size=(4, 6))

    np.save(npyfile, data)
    with open(npyfile, mode=mode) as f:
        fileview = make_mpi_file_view(f, comm=comm_self)
        read_data = fileview.read()
        np.testing.assert_allclose(read_data, data)

        # assert data can be read several times
        read_data = fileview.read()
        np.testing.assert_allclose(read_data, data)


def test_filesave_3d(comm, globaldata3d):
    distdata = subdivide(comm, globaldata3d)

    save_npy(
        "test3d.npy",
        distdata.data,
        distdata.subdomain_locations,
        distdata.nb_domain_grid_pts,
        comm,
    )
    comm.barrier()
    if comm.Get_rank() == 0:
        loaded_data = np.load("test3d.npy")
        np.testing.assert_array_equal(loaded_data, globaldata3d)

        os.remove("test3d.npy")
    comm.barrier()
