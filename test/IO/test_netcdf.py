#
# Copyright 2021 Antoine Sanner
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

# This was migrated from muSpectre by Antoine Sanner. Lars Pastewka is author.

import numpy as np
import pytest

try:
    # raise ImportError()
    from netCDF4 import Dataset

    _has_netCDF4 = True
except ImportError:
    _has_netCDF4 = False
###

if _has_netCDF4:
    from NuMPI.IO.NetCDF import NCStructuredGrid
else:
    pytestmark = pytest.mark.skip("skip because no NetCDF installed")


class DomainDecomposition():
    def __init__(self, nb_domain_grid_pts, communicator):
        nprocs = communicator.Get_size()
        rank = communicator.Get_rank()

        self.subdomain_slices = [slice(None), ] * len(nb_domain_grid_pts)
        self.subdomain_locations = [0, ] * len(nb_domain_grid_pts)
        self.nb_subdomain_grid_pts = list(nb_domain_grid_pts)

        step = nb_domain_grid_pts[0] // nprocs
        # decomposition along first axis
        if rank == nprocs - 1:
            self.subdomain_slices[0] = slice(rank * step, None)
            self.subdomain_locations[0] = rank * step
            self.nb_subdomain_grid_pts[0] = nb_domain_grid_pts[0] - rank * step
        else:
            self.subdomain_slices[0] = slice(rank * step, (rank + 1) * step)
            self.subdomain_locations[0] = rank * step
            self.nb_subdomain_grid_pts[0] = step


@pytest.fixture(params=[(11,), (11, 23), (11, 23, 7)])
def self(request, comm):
    # This fixture mimics the setup in the unit test class.
    # This test was written in the unittest syntax before.

    class self:
        pass

    self.nb_grid_pts = request.param
    self.tensor_shape = tuple(list(self.nb_grid_pts) + [3, 3])
    self.scalar_grid = np.arange(np.prod(self.nb_grid_pts)).reshape(self.nb_grid_pts)
    self.tensor_grid = np.arange(np.prod(self.tensor_shape)).reshape(self.tensor_shape)

    self.communicator = comm

    self.domain_decomposition = DomainDecomposition(self.nb_grid_pts, self.communicator)

    return self


def test_write_read_domain(self):
    if self.communicator is None:
        nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                              nb_domain_grid_pts=self.nb_grid_pts)
    else:
        nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                              nb_domain_grid_pts=self.nb_grid_pts,
                              decomposition='domain',
                              subdomain_locations=self.domain_decomposition.subdomain_locations,
                              nb_subdomain_grid_pts=self.domain_decomposition.nb_subdomain_grid_pts,
                              communicator=self.communicator)
    nc.scalar = self.scalar_grid
    nc.tensor = self.tensor_grid
    nc[3].per_frame_tensor = self.tensor_grid
    nc.close()
    self.communicator.barrier()
    # Check that the file structure is correct
    nc = Dataset('test_{}d.nc'.format(len(self.nb_grid_pts)), 'r', comm=None)
    dimensions = ['frame', 'grid_x', 'tensor_3']
    if len(self.nb_grid_pts) > 1:
        dimensions += ['grid_y']
    if len(self.nb_grid_pts) == 3:
        dimensions += ['grid_z']
    assert set(nc.dimensions) == set(dimensions)
    assert len(nc.dimensions['frame']) == 4
    assert len(nc.dimensions['grid_x']) == self.nb_grid_pts[0]
    if len(self.nb_grid_pts) > 1:
        assert len(nc.dimensions['grid_y']) == self.nb_grid_pts[1]
    if len(self.nb_grid_pts) == 3:
        assert len(nc.dimensions['grid_z']) == self.nb_grid_pts[2]
    assert len(nc.dimensions['tensor_3']) == 3
    nc.close()

    # Read file and check data
    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='r', communicator=None)
    assert np.equal(tuple(nc.nb_domain_grid_pts), tuple(self.nb_grid_pts)).all()
    assert np.equal(nc.scalar, self.scalar_grid).all()
    assert np.equal(nc.tensor, self.tensor_grid).all()
    assert np.equal(nc[3].per_frame_tensor, self.tensor_grid).all()
    nc.close()


def test_write_read_subdomain(self):
    scalar_grid = self.scalar_grid[tuple(self.domain_decomposition.subdomain_slices)]
    tensor_grid = self.tensor_grid[tuple(self.domain_decomposition.subdomain_slices)]

    if self.communicator is None:
        nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                              nb_domain_grid_pts=self.nb_grid_pts)
    else:
        nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                              nb_domain_grid_pts=self.nb_grid_pts,
                              decomposition='subdomain',
                              subdomain_locations=self.domain_decomposition.subdomain_locations,
                              nb_subdomain_grid_pts=self.domain_decomposition.nb_subdomain_grid_pts,
                              communicator=self.communicator)
    nc.scalar = scalar_grid
    nc.tensor = tensor_grid
    nc[3].per_frame_tensor = tensor_grid
    nc.close()
    self.communicator.barrier()
    # Check that the file structure is correct
    nc = Dataset('test_{}d.nc'.format(len(self.nb_grid_pts)), 'r', comm=None)
    dimensions = ['frame', 'grid_x', 'tensor_3']
    if len(self.nb_grid_pts) > 1:
        dimensions += ['grid_y']
    if len(self.nb_grid_pts) == 3:
        dimensions += ['grid_z']
    assert np.equal(set(nc.dimensions), set(dimensions))
    assert len(nc.dimensions['frame']) == 4
    assert len(nc.dimensions['grid_x']) == self.nb_grid_pts[0]
    if len(self.nb_grid_pts) > 1:
        assert len(nc.dimensions['grid_y']) == self.nb_grid_pts[1]
    if len(self.nb_grid_pts) == 3:
        assert len(nc.dimensions['grid_z']) == self.nb_grid_pts[2]
    assert len(nc.dimensions['tensor_3']) == 3
    nc.close()

    # Read file and check data
    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='r', communicator=None)
    assert np.equal(tuple(nc.nb_domain_grid_pts), tuple(self.nb_grid_pts)).all()
    assert np.equal(nc.scalar, self.scalar_grid).all()
    assert np.equal(nc.tensor, self.tensor_grid).all()
    assert np.equal(nc[3].per_frame_tensor, self.tensor_grid).all()
    nc.close()
