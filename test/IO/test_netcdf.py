"""

Licence HEADER from muspectre

@author Till Junge <till.junge@altermail.ch>

@date   17 Jan 2018

Copyright © 2018 Till Junge

µFFT is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µFFT is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µFFT; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

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


# @pytest.mark.parametrize("nb_grid_pts", [(11, 23), (11, 23, 7)])
@pytest.fixture(params=[(11,), (11, 23), (11, 23, 7)])
def self(request, comm):
    class self:
        pass

    self.nb_grid_pts = request.param
    self.tensor_shape = tuple(list(self.nb_grid_pts) + [3, 3])
    self.scalar_grid = np.arange(np.prod(self.nb_grid_pts)).reshape(self.nb_grid_pts)
    self.tensor_grid = np.arange(np.prod(self.tensor_shape)) \
        .reshape(self.tensor_shape)

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

    # Check that the file structure is correct
    nc = Dataset('test_{}d.nc'.format(len(self.nb_grid_pts)), 'r')
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
    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='r')
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

    # Check that the file structure is correct
    nc = Dataset('test_{}d.nc'.format(len(self.nb_grid_pts)), 'r')
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
    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='r')
    assert np.equal(tuple(nc.nb_domain_grid_pts), tuple(self.nb_grid_pts)).all()
    assert np.equal(nc.scalar, self.scalar_grid).all()
    assert np.equal(nc.tensor, self.tensor_grid).all()
    assert np.equal(nc[3].per_frame_tensor, self.tensor_grid).all()
    nc.close()
