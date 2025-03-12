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

import struct
from ast import literal_eval
from itertools import product

import numpy as np
from numpy.lib.format import MAGIC_PREFIX, _filter_header, magic

from .. import MPI
from .common import MPIFileTypeError, MPIFileView


def _chunked_read_write(
    file,
    chunk_op,
    header_len,
    nb_grid_pts,
    nb_subdomain_grid_pts,
    subdomain_locations,
    fortran_order,
    data,
    comm,
):
    nb_dims = len(nb_grid_pts)
    if fortran_order:
        # The data in the file is in Fortran order.
        # We flip the axes for reading and writing.
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

    nb_max_subdomain_grid_pts = np.empty_like(nb_subdomain_grid_pts)
    comm.Allreduce(
        np.array(nb_subdomain_grid_pts, order="C"),
        nb_max_subdomain_grid_pts,
        op=MPI.MAX,
    )

    for subdomain_coords in product(
        *(range(nb_max_subdomain_grid_pts[axis]) for axis in axes[:-2])
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
        if chunk_op == "read":
            if fortran_order:
                chunk = np.empty(
                    (nb_subdomain_grid_pts[1], nb_subdomain_grid_pts[0]),
                    dtype=data.dtype,
                )
            else:
                try:
                    chunk = data[subdomain_coords]
                except IndexError:
                    chunk = np.empty((0,), dtype=data.dtype)  # Nothing to read
            file.Read_all(chunk)
            if fortran_order:
                try:
                    data.T[subdomain_coords] = chunk
                except IndexError:
                    pass
        elif chunk_op == "write":
            try:
                if fortran_order:
                    chunk = data.T[subdomain_coords]
                else:
                    chunk = data[subdomain_coords]
            except IndexError:
                chunk = np.empty((0,), dtype=data.dtype)  # Nothing to write
            file.Write_all(chunk)

        else:
            raise ValueError("Unknown chunk operation '{}'".format(chunk_op))

    filetype.Free()


class NPYFile(MPIFileView):
    """
    You may have a look at numpy.lib.format if you want to understand how
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
            self.file = MPI.File.Open(self.comm, self.fn, MPI.MODE_RDONLY)
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
            self.header_length = struct.unpack(hlength_type, hlength_str)[0]
            header = mpi_read_bytes(self.file, self.header_length)

            header = _filter_header(header.decode("latin1"))
            d = literal_eval(header)  # TODO: Copy from _read_array_header  with all the
            # assertions
            self.dtype = np.dtype(d["descr"])
            self.fortran_order = d["fortran_order"]
            self.nb_grid_pts = d["shape"]
            self.data_start = self.file.Get_position()

            self.header_length += len(magic_str) + struct.calcsize(hlength_type)
        except Exception as err:
            # FIXME! This should be handled through a resource manager
            if self.file is not None and self.close_file_on_error:
                self.file.Close()
            raise err

    def read(self, subdomain_locations=None, nb_subdomain_grid_pts=None):
        nb_dims = len(self.nb_grid_pts)
        if subdomain_locations is None:
            subdomain_locations = (0,) * nb_dims
        if nb_subdomain_grid_pts is None:
            nb_subdomain_grid_pts = self.nb_grid_pts

        data = np.empty(nb_subdomain_grid_pts, dtype=self.dtype)

        _chunked_read_write(
            self.file,
            "read",
            self.header_length,
            self.nb_grid_pts,
            nb_subdomain_grid_pts,
            subdomain_locations,
            self.fortran_order,
            data,
            self.comm,
        )

        return data

    def close(self):
        self.file.Close()


def mpi_open(fn, comm, format=None):
    readers = {"npy": NPYFile}

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
    header_length = len(arr_dict_str) + len(magic_str) + 2

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode("latin-1"))

    _chunked_read_write(
        file,
        "write",
        header_length,
        nb_grid_pts,
        nb_subdomain_grid_pts,
        subdomain_locations,
        data.flags.f_contiguous,
        data,
        comm,
    )

    file.Close()


def load_npy(
    fn, subdomain_locations=None, nb_subdomain_grid_pts=None, comm=MPI.COMM_WORLD
):
    file = NPYFile(fn, comm)
    data = file.read(subdomain_locations, nb_subdomain_grid_pts)
    file.close()
    return data
