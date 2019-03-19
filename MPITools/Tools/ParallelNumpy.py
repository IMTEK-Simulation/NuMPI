
import numpy as np
from mpi4py import MPI

def get_dtypeInfo(dtype):
    if dtype.kind == 'i': return np.iinfo(dtype)
    if dtype.kind == 'f': return np.finfo(dtype)
    raise ValueError


class ParallelNumpy :
    def __init__(self,comm=MPI.COMM_WORLD):
        self.comm = comm

    def sum(self,arr,*args,**kwargs):
        """
        take care that the input arrays have the same datatype on all Processors !

        when you specify an axis along which make the sum, be shure that it is the direction in which data is decomposed (for slab data decomposition) !

        pencil data decomposition is not implemented yet.

        Parameters
        ----------
        arr: numpy Array

        Returns
        -------
        scalar np.ndarray , the sum of all Elements of the Array over all the Processors
        """

        locresult= np.sum(arr,*args,**kwargs)
        result = np.zeros_like(locresult)
        #print("Proc{}: result.dtype={}, locresult.dtype={} arr.dtype={}".format(self.comm.Get_rank(),result.dtype,locresult.dtype,arr.dtype))
        mpitype = MPI._typedict[locresult.dtype.char]
        self.comm.Allreduce([locresult,mpitype],[result,mpitype],op = MPI.SUM)
        if type(locresult) == type(result):#TODO:Why is this done again ? 
            return result
        else :
            return type(locresult)(result)

    def max(self,arr):
        """
        take care that the input arrays have the same datatype on all Processors !
        Parameters
        ----------
        arr: numpy float array, can be empty

        Returns
        -------
        np.array of size 1, the max value of arr over all arrays, if all are empty this is -np.inf

        """
        result = np.asarray(0, dtype=arr.dtype)

        absmin = get_dtypeInfo(arr.dtype).min # most negative value that can be stored in this datatype
        mpitype = MPI._typedict[arr.dtype.char]
        self.comm.Allreduce([np.max(arr) if arr.size > 0 else np.array([absmin],dtype=arr.dtype),mpitype], [result,mpitype], op=MPI.MAX)
        # FIXME: use the max of the dtype because np.inf only float
        #TODO: Not elegant, but following options didn't work
        #self.comm.Allreduce(np.max(arr) if arr.size > 0 else np.array(None, dtype=arr.dtype), result, op=MPI.MAX)
        # Here when the first array isn't empty it is fine, but otherwise the result will be nan
        #
        #self.comm.Allreduce(np.max(arr) if arr.size > 0 else np.array([], dtype=arr.dtype), result, op=MPI.MAX)
        # Her MPI claims that the input and output array have not the same datatype
        return result

    def min(self,arr):
        """
        take care that the input arrays have the same datatype on all Processors !
        Parameters
        ----------
        arr: numpy float array, can be empty

        Returns
        -------
        np.array of size 1, the min value of arr over all arrays, if all are empty this is np.inf

        """
        result = np.asarray(0, dtype=arr.dtype)
        absmax = get_dtypeInfo(arr.dtype).max # most positive value that can be stored in this datatype
        mpitype = MPI._typedict[arr.dtype.char]
        self.comm.Allreduce([np.min(arr) if arr.size > 0 else np.array([absmax],dtype=arr.dtype),mpitype] ,[ result,mpitype], op=MPI.MIN)
        return result

    def dot(self,a,b):
        locresult = np.dot(a,b)
        result = np.zeros_like(locresult)
        mpitype = MPI._typedict[locresult.dtype.char]
        self.comm.Allreduce([locresult,mpitype], [result, mpitype], op=MPI.SUM)
        return result

    def any(self, arr):
        result = np.array(False, dtype=bool)
        self.comm.Allreduce(np.any(arr), result, op=MPI.LOR)
        return result.item()

    def all(self, arr):
        result = np.array(False, dtype=bool)
        self.comm.Allreduce(np.all(arr), result, op=MPI.LAND)
        return result.item()