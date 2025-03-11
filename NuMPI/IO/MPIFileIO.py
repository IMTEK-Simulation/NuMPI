#
# Copyright 2018, 2020 Antoine Sanner
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


"""
MPI-parallel writing of arrays in numpy's 'npy' format.
"""

import abc
import struct
from ast import literal_eval
from itertools import product

import numpy as np
from numpy.lib.format import MAGIC_PREFIX, _filter_header, magic

from .. import MPI


class MPIFileTypeError(Exception):
    pass


class MPIFileIncompatibleResolutionError(Exception):
    pass


class MPIFileView(metaclass=abc.ABCMeta):
    def __init__(self, fn, comm):
        self.fn = fn
        self.comm = comm
        # if hasattr read, it is a stream and it should not close the file
        self.close_file_on_error = not hasattr(fn, "read")

    @abc.abstractmethod
    def _read_header(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass


class MPIFileViewNPY(MPIFileView):
    """

    you may have a look at numpy.lib.format if you want to understand how
    this code works
    """

    def __init__(self, fn, comm):
        super().__init__(fn, comm)
        self._read_header()

    def detect_format(self):  # TODO: maybe useless
        try:
            self._read_header()
            return True
        except Exception:
            return False

    def _read_header(self):
        self.file = None
        try:
            magic_str = magic(1, 0)
            self.file = MPI.File.Open(self.comm, self.fn, MPI.MODE_RDONLY)  #
            magic_str = mpi_read_bytes(self.file, len(magic_str))
            if magic_str[:-2] != MAGIC_PREFIX:
                raise MPIFileTypeError(
                    "MAGIC_PREFIX missing at the beginning of file {}".format(self.fn)
                )

            version = magic_str[-2:]

            if version == b"\x01\x00":
                hlength_type = "<H"
            elif version == b"\x02\x00":
                hlength_type = "<I"
            else:
                raise MPIFileTypeError("Invalid version %r" % version)

            hlength_str = mpi_read_bytes(self.file, struct.calcsize(hlength_type))
            header_length = struct.unpack(hlength_type, hlength_str)[0]
            header = mpi_read_bytes(self.file, header_length)

            header = _filter_header(header.decode("latin1"))
            d = literal_eval(header)  # TODO: Copy from _read_array_header  with all the
            # assertions
            self.dtype = np.dtype(d["descr"])
            self.fortran_order = d["fortran_order"]
            self.nb_grid_pts = d["shape"]
            self.data_start = self.file.Get_position()
        except Exception as err:
            # FIXME! This should be handled through a resource manager
            if self.file is not None and self.close_file_on_error:
                self.file.Close()
            raise err

    def read(self, subdomain_locations=None, nb_subdomain_grid_pts=None):

        if subdomain_locations is None:
            subdomain_locations = (0, 0)
        if nb_subdomain_grid_pts is None:
            nb_subdomain_grid_pts = self.nb_grid_pts

        # Now how to start reading ?

        mpitype = MPI._typedict[self.dtype.char]

        if self.fortran_order:
            # the returned array will be in fortran_order.
            # the data is loaded in C_contiguous array but in a transposed
            # manner
            # data.transpose() is called which swaps the shapes back again and
            # toggles C-order to F-order
            ix = 1
            iy = 0
        else:
            ix = 0
            iy = 1

        # create a type
        filetype = mpitype.Create_vector(
            nb_subdomain_grid_pts[ix],
            # number of blocks  : length of data in the non-contiguous
            # direction
            nb_subdomain_grid_pts[iy],
            # length of block : length of data in contiguous direction
            self.nb_grid_pts[iy],
            # stepsize: the data is contiguous in y direction,
            # two matrix elements with same x position are separated by ny
            # in memory
        )

        filetype.Commit()  # verification if type is OK
        self.file.Set_view(
            self.data_start
            + (subdomain_locations[ix] * self.nb_grid_pts[iy] + subdomain_locations[iy])
            * mpitype.Get_size(),
            filetype=filetype,
        )

        data = np.empty(
            (nb_subdomain_grid_pts[ix], nb_subdomain_grid_pts[iy]), dtype=self.dtype
        )

        self.file.Read_all(data)
        filetype.Free()

        if self.fortran_order:
            data = data.transpose()

        return data

    def close(self):
        self.file.Close()


def make_mpi_file_view(fn, comm, format=None):  # TODO: DISCUSS: oder als __init__ von
    # der MPIFileView Klasse ?
    readers = {"npy": MPIFileViewNPY}

    if format is not None:
        try:
            reader = readers[format]
        except KeyError:
            raise (
                ValueError(
                    "Given format is not recognised, you should give {}".format(
                        readers.keys()
                    )
                )
            )
        return reader(fn, comm)

    for reader in readers.values():
        try:
            return reader(fn, comm)
        except MPIFileTypeError:
            pass
    raise MPIFileTypeError(
        "No MPI filereader was able to open_topography the file {}".format(fn)
    )


def mpi_read_bytes(file, nbytes):
    # allocate the buffer
    buf = np.empty(nbytes, dtype=np.int8)
    file.Read_all(buf)
    return buf.tobytes()


def save_npy(fn, data, subdomain_locations=None, nb_grid_pts=None, comm=MPI.COMM_WORLD):
    """

    Parameters
    ----------
    data : numpy array : data owned by the processor
    location : index of the first element of data within the global data
    nb_grid_pts : nb_grid_pts of the global data
    comm : MPI communicator

    Returns
    -------

    """
    data = np.asarray(data)

    if not data.flags.f_contiguous and not data.flags.c_contiguous:
        raise ValueError("Data must be contiguous")

    nb_dims = len(data.shape)
    assert (
        len(nb_grid_pts) == nb_dims
    ), "`nb_grid_pts` must have the same number of dimensions as the data`"

    if subdomain_locations is None:
        subdomain_locations = (0,) * nb_dims
    else:
        assert (
            len(subdomain_locations) == nb_dims
        ), "`subdomain_locations` must have the same number of dimensions as the data`"
    nb_subdomain_grid_pts = data.shape

    from numpy.lib.format import dtype_to_descr, magic

    magic_str = magic(1, 0)

    arr_dict_str = str(
        {
            "descr": dtype_to_descr(data.dtype),
            "fortran_order": data.flags.f_contiguous,
            "shape": tuple(nb_grid_pts),
        }
    )

    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += " "
    arr_dict_str += "\n"
    header_len = len(arr_dict_str) + len(magic_str) + 2

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode("latin-1"))

    if data.flags.f_contiguous:
        # the returned array will be in fortran_order.
        # the data is loaded in C_contiguous array but in a transposed manner
        # data.transpose() is called which swaps the shapes back again and
        # toggles C-order to F-order
        axes = tuple(range(nb_dims - 1, -1, -1))
    else:
        axes = tuple(range(nb_dims))

    mpitype = MPI._typedict[data.dtype.char]
    # see MPI_TYPE_VECTOR
    filetype = mpitype.Create_vector(
        # number of blocks: length of data in the non-contiguous direction
        nb_subdomain_grid_pts[axes[-2]] if nb_dims > 1 else 1,
        # length of block: length of data in contiguous direction
        nb_subdomain_grid_pts[axes[-1]],
        # stride: the data is contiguous in z direction,
        # two matrix elements with same x position are separated by ny*nz in memory
        nb_grid_pts[axes[-1]],
    )  # create a type
    filetype.Commit()  # verification if type is OK

    for subdomain_coords in product(
        *(range(nb_subdomain_grid_pts[axis]) for axis in axes[:-2])
    ):
        offset = 0
        for axis, coord in zip(axes[:-2], subdomain_coords):
            offset = offset * nb_grid_pts[axis] + subdomain_locations[axis] + coord
        for axis in axes[-2:]:
            offset = offset * nb_grid_pts[axis] + subdomain_locations[axis]

        file.Set_view(
            header_len + offset * mpitype.Get_size(),
            filetype=filetype,
        )
        if data.flags.f_contiguous:
            chunk = data.T[subdomain_coords]
        else:
            chunk = data[subdomain_coords]
        file.Write_all(chunk)

    filetype.Free()

    file.Close()


def load_npy(
    fn, subdomain_locations=None, nb_subdomain_grid_pts=None, comm=MPI.COMM_WORLD
):
    file = MPIFileViewNPY(fn, comm)
    # if file.nb_grid_pts != nb_domain_grid_pts:
    #    raise MPIFileIncompatibleResolutionError(
    #        "nb_domain_grid_pts is {} but file nb_grid_pts is {}".format(
    #        nb_domain_grid_pts, file.nb_grid_pts))
    data = file.read(subdomain_locations, nb_subdomain_grid_pts)
    file.close()
    return data
