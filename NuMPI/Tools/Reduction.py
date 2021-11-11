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


def get_dtype_info(dtype):
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

    def _op(self, npop, npargs, mpiop, *args, **kwargs):
        """
        Generic reduction operation

        Parameters
        ----------
        npop : func
            Numpy reduction function (e.g. np.sum)
        npargs: tuple
            Arguments passed to the reduction function (e.g. array to be
            reduced)
        mpiop : mpi4py.MPI.op
            MPI reduction operation

        Returns
        -------
        result_arr : np.ndarray
            Result of the reduction operation
        """
        local_result = npop(*npargs, *args, **kwargs)
        result = np.zeros_like(local_result)
        mpitype = MPI._typedict[local_result.dtype.char]
        self.comm.Allreduce([local_result, mpitype], [result, mpitype], op=mpiop)
        return result

    def _op1(self, npop, arr, mpiop, *args, **kwargs):
        """
        Generic reduction operation that takes a single (array) argument

        Parameters
        ----------
        npop : func
            Numpy reduction function (e.g. np.sum)
        arr : array_like
            Numpy array containing the data to be reduced
        mpiop : mpi4py.MPI.op
            MPI reduction operation
        initial : arr.dtype
            Value to use if local array is empty

        Returns
        -------
        result_arr : np.ndarray
            Result of the reduction operation
        """
        if 'initial' in kwargs and isinstance(arr, np.ma.MaskedArray):
            # Max/min on masked array do not support `initial`
            arr = arr.filled(kwargs['initial'])
            del kwargs['initial']
        return self._op(npop, (arr,), mpiop, *args, **kwargs)

    def sum(self, arr, *args, **kwargs):
        """
        Summation

        Parameters
        ----------
        arr : array_like
            Numpy array containing the data to be reduced

        Returns
        -------
        result_arr : np.ndarray
            Sum of all elements of the array over all processors
        """
        return self._op1(np.sum, arr, MPI.SUM, *args, **kwargs)

    def max(self, arr, *args, **kwargs):
        """
        Maximum value

        Parameters
        ----------
        arr : array_like
            Numpy array containing the data to be reduced

        Returns
        -------
        result_arr : np.ndarray
            Maximum of all elements of the array over all processors
        """
        kwargs['initial'] = get_dtype_info(arr.dtype).min
        return self._op1(np.max, arr, MPI.MAX, *args, **kwargs)

    def min(self, arr, *args, **kwargs):
        """
        Minimum value

        Parameters
        ----------
        arr : array_like
            Numpy array containing the data to be reduced

        Returns
        -------
        result_arr : np.ndarray
            Minimum of all elements of the array over all processors
        """
        kwargs['initial'] = get_dtype_info(arr.dtype).max
        return self._op1(np.min, arr, MPI.MIN, *args, **kwargs)

    def mean(self, arr, *args, **kwargs):
        """
        Arithmetic mean

        Parameters
        ----------
        arr : array_like
            Numpy array containing the data to be reduced

        Returns
        -------
        result_arr : np.ndarray
            Arithmetic mean of all elements of the array over all processors
        """
        return self.sum(arr, *args, **kwargs) / self.sum(np.ones_like(arr), *args, **kwargs)

    def dot(self, a, b, *args, **kwargs):
        """
        Scalar product a.b

        Parameters
        ----------
        a : array_like
            Numpy array containing the data of the first array
        a : array_like
            Numpy array containing the data of the second array

        Returns
        -------
        result_arr : np.ndarray
            Scalar product between a and b
        """
        return self._op(np.dot, (a, b), MPI.SUM, *args, **kwargs)

    def any(self, arr, *args, **kwargs):
        """
        Returns true of any value is true

        Parameters
        ----------
        arr : array of bools
            Input data

        Returns
        -------
        result_arr : np.ndarray
            True if any value in `arr` is true
        """
        return self._op1(np.any, arr, MPI.LOR, *args, **kwargs)

    def all(self, arr, *args, **kwargs):
        """
        Returns true of all values are true

        Parameters
        ----------
        arr : array of bools
            Input data

        Returns
        -------
        result_arr : np.ndarray
            True if all values in `arr` are true
        """
        return self._op1(np.all, arr, MPI.LAND, *args, **kwargs)
