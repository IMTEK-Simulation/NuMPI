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

import numpy as np

from .. import MPI


def get_dtypeInfo(dtype):
    if dtype.kind == 'i':
        return np.iinfo(dtype)
    if dtype.kind == 'f':
        return np.finfo(dtype)
    raise ValueError


class Reduction:
    def __init__(self, comm=None):
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm

    def sum(self, arr, *args, **kwargs):
        """
        take care that the input arrays have the same datatype on all
        Processors !

        when you specify an axis along which make the sum, be shure that it
        is the direction in which data is decomposed (for slab data
        decomposition) !

        pencil data decomposition is not implemented yet.

        Parameters
        ----------
        arr: numpy Array

        Returns
        -------
        scalar np.ndarray , the sum of all Elements of the Array over all
        the Processors
        """

        locresult = np.sum(arr, *args, **kwargs)
        result = np.zeros_like(locresult)
        # print("Proc{}: result.dtype={}, locresult.dtype={} arr.dtype={
        # }".format(self.comm.Get_rank(),result.dtype,locresult.dtype,
        # arr.dtype))
        mpitype = MPI._typedict[locresult.dtype.char]
        self.comm.Allreduce([locresult, mpitype], [result, mpitype],
                            op=MPI.SUM)
        if type(locresult) == type(result):  # TODO:Why is this done again ?
            return result
        else:
            return type(locresult)(result)

    def max(self, arr):
        """
        take care that the input arrays have the same datatype on all
        Processors !
        Parameters
        ----------
        arr: numpy float array, can be empty

        Returns
        -------
        np.array of size 1, the max value of arr over all arrays, if all are
        empty this is -np.inf

        """
        result = np.asarray(0, dtype=arr.dtype)

        absmin = get_dtypeInfo(arr.dtype).min  # most negative value that can
        #                                        be stored in this datatype
        mpitype = MPI._typedict[arr.dtype.char]
        self.comm.Allreduce([np.max(arr) if arr.size > 0 else np.array(
            [absmin], dtype=arr.dtype), mpitype],
                            [result, mpitype], op=MPI.MAX)
        # FIXME: use the max of the dtype because np.inf only float
        # TODO: Not elegant, but following options didn't work
        # self.comm.Allreduce(np.max(arr) if arr.size > 0 else np.array(
        # None, dtype=arr.dtype), result, op=MPI.MAX)
        # Here when the first array isn't empty it is fine, but otherwise
        # the result will be nan
        #
        # self.comm.Allreduce(np.max(arr) if arr.size > 0 else np.array([],
        # dtype=arr.dtype), result, op=MPI.MAX)
        # Her MPI claims that the input and output array have not the same
        # datatype
        return result

    def min(self, arr):
        """
        take care that the input arrays have the same datatype on all
        Processors !
        Parameters
        ----------
        arr: numpy float array, can be empty

        Returns
        -------
        np.array of size 1, the min value of arr over all arrays, if all are
        empty this is np.inf

        """
        result = np.asarray(0, dtype=arr.dtype)
        absmax = get_dtypeInfo(arr.dtype).max  # most positive value that can
        #                                        be stored in this datatype
        mpitype = MPI._typedict[arr.dtype.char]
        self.comm.Allreduce([np.min(arr) if arr.size > 0 else np.array(
            [absmax], dtype=arr.dtype), mpitype],
                            [result, mpitype], op=MPI.MIN)
        return result

    def dot(self, a, b):
        locresult = np.dot(a, b)
        result = np.zeros_like(locresult)
        mpitype = MPI._typedict[locresult.dtype.char]
        self.comm.Allreduce([locresult, mpitype], [result, mpitype],
                            op=MPI.SUM)
        return result

    def any(self, arr):
        result = np.array(False, dtype=bool)
        self.comm.Allreduce(np.array(np.any(arr)), result, op=MPI.LOR)
        return result.item()

    def all(self, arr):
        result = np.array(False, dtype=bool)
        self.comm.Allreduce(np.array(np.all(arr), dtype=bool), result,
                            op=MPI.LAND)
        return result.item()
