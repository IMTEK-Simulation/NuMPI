
import unittest
import numpy as np

from MPITools import MPI
from MPITools.Tools import ParallelNumpy

class test_ParallelNumpy(unittest.TestCase):

    def setUp(self):
        self.np = ParallelNumpy()
        self.comm = MPI.COMM_WORLD
        self.rank  = self.comm.Get_rank()
        self.MPIsize = self.comm.Get_size()
    def test_sum_scalar(self):
        res=self.np.sum(np.array(1))
        self.assertEqual(res, self.np.comm.Get_size())

    def test_sum_1D(self):
        arr=np.array((1,2.1,3))
        res = self.np.sum(arr)
        np.testing.assert_allclose(res, self.np.comm.Get_size() * 6.1,atol=1e-12)

    def test_sum_2D(self):
        arr=np.array(((1,2.1,3),
                     (4,5,6)))
        res = self.np.sum(arr)
        np.testing.assert_allclose(res, self.np.comm.Get_size() * 21.1,atol=1e-12)

    def test_sum_boolean(self):
        arr=np.array(((1,2.1,3),
                     (4,5,6)))
        arr=arr>3

        #print(arr.dtype)
        res = self.np.sum(arr)
        self.assertEqual(res,self.np.comm.Get_size()*3)

    def test_sum_along_axis_decomp(self):
        domain_resolution = (128, 65)
        np.random.seed(2)
        globaldata = np.random.random(domain_resolution)

        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        step = domain_resolution[0] // nprocs

        if rank == nprocs - 1:
            subdomain_slice = (slice(rank * step, None), slice(None, None))
            subdomain_location = [rank * step, 0]
            subdomain_resolution = [domain_resolution[0] - rank * step, domain_resolution[1]]
        else:
            subdomain_slice = (slice(rank * step, (rank + 1) * step), slice(None, None))
            subdomain_location = [rank * step, 0]
            subdomain_resolution = [step, domain_resolution[1]]

        localdata = globaldata[subdomain_slice]

        np.testing.assert_allclose(self.np.sum(localdata,axis=0), np.sum(globaldata,axis=0),rtol = 1e-12)
        # Why are they not perfectly equal ?
    def test_max_2D(self):
        arr=np.reshape(np.array((-1,1,5,4,
                             4,5,4,5,
                             7,0,1,0.),dtype=float),(3,4))

        rank = self.comm.Get_rank()
        if self.comm.Get_size() >=4:
            if rank ==0 :   local_arr = arr[0:2,0:2]
            elif rank ==1 : local_arr = arr[0:2,2:]
            elif rank == 2 :local_arr = arr[2:,0:2]
            elif rank == 3 : local_arr = arr[2:,2:]
            else : local_arr = np.empty(0,dtype=arr.dtype)
        elif self.comm.Get_size() >=2 :
            if   rank ==0 :   local_arr = arr[0:2,:]
            elif rank ==1 : local_arr = arr[2:,:]
            else:           local_arr = np.empty(0, dtype=arr.dtype)
        else:
            local_arr = arr
        self.assertEqual(self.np.max(local_arr),7)


    def test_max_2D_int(self):
        arr=np.reshape(np.array((-1,1,5,4,
                             4,5,4,5,
                             7,0,1,0),dtype=int),(3,4))

        rank = self.comm.Get_rank()
        if self.comm.Get_size() >=4:
            if rank ==0 :   local_arr = arr[0:2,0:2]
            elif rank ==1 : local_arr = arr[0:2,2:]
            elif rank == 2 :local_arr = arr[2:,0:2]
            elif rank == 3 : local_arr = arr[2:,2:]
            else : local_arr = np.empty(0,dtype=arr.dtype)
        elif self.comm.Get_size() >=2 :
            if   rank ==0 :   local_arr = arr[0:2,:]
            elif rank ==1 : local_arr = arr[2:,:]
            else:           local_arr = np.empty(0, dtype=arr.dtype)
        else:
            local_arr = arr
        self.assertEqual(self.np.max(local_arr),7)

    def test_max_min_empty(self):
        """
        Sometimes the input array is empty
        """
        if self.MPIsize >=2 :
            if self.rank==0:
                local_arr = np.array([], dtype=float)

            else :
                local_arr = np.array([1, 0, 4], dtype=float)
            self.assertEqual(self.np.max(local_arr), 4)
            self.assertEqual(self.np.min(local_arr), 0)

            if self.rank==0:
                local_arr = np.array([1, 0, 4], dtype=float)
            else :

                local_arr = np.array([], dtype=float)
            self.assertEqual(self.np.max(local_arr), 4)
            self.assertEqual(self.np.min(local_arr), 0)

        else :
            local_arr = np.array([],dtype = float)
            #self.assertTrue(np.isnan(self.np.max(local_arr)))
            #self.assertTrue(np.isnan(self.np.min(local_arr)))
            self.assertEqual(self.np.max(local_arr),np.finfo(local_arr.dtype).min)
            self.assertEqual(self.np.min(local_arr),np.finfo(local_arr.dtype).max)



    def test_min(self):
        arr = np.reshape(np.array((-1, 1, 5, 4,
                                   4, 5, 4, 5,
                                   7, 0, 1, 0),dtype = float), (3, 4))

        rank = self.comm.Get_rank()
        if self.comm.Get_size() >= 4:
            if rank == 0:
                local_arr = arr[0:2, 0:2]
            elif rank == 1:
                local_arr = arr[0:2, 2:]
            elif rank == 2:
                local_arr = arr[2:, 0:2]
            elif rank == 3:
                local_arr = arr[2:, 2:]
            else:
                local_arr = np.empty(0, dtype=arr.dtype)
        elif self.comm.Get_size() >= 2:
            if rank == 0:
                local_arr = arr[0:2, :]
            elif rank == 1:
                local_arr = arr[2:, :]
            else:
                local_arr = np.empty(0, dtype=arr.dtype)
        else:
            local_arr = arr
        self.assertEqual(self.np.min(local_arr), -1)

    def test_dot_row_vectors(self):
        np.random.seed(1)

        n = 10
        fulla = np.random.random(n)
        fullb = np.random.random(n)

        step = n // self.MPIsize

        if self.rank == self.MPIsize - 1:
            loc_sl = slice(self.rank * step, None)
        else:
            loc_sl  = slice(self.rank * step, (self.rank + 1) * step)


        self.assertAlmostEqual(np.asscalar(self.np.dot(fulla[loc_sl],fullb[loc_sl])), np.dot(fulla, fullb),7)


    def test_dot_matrix_vector(self):
        np.random.seed(1)

        m=13
        n = 11

        step = n // self.MPIsize

        if self.rank == self.MPIsize - 1:
            loc_sl = slice(self.rank * step, None)
        else:
            loc_sl = slice(self.rank * step, (self.rank + 1) * step)

        fulla = np.random.random((m,n))
        for fullb in [np.random.random((n,1)),np.random.random(n)]:
            np.testing.assert_allclose(self.np.dot(fulla[:,loc_sl], fullb[loc_sl]), np.dot(fulla, fullb))
            np.testing.assert_allclose(self.np.dot(fullb[loc_sl].T,fulla[:,loc_sl].T),np.dot(fullb.T, fulla.T))
            with self.assertRaises(ValueError):
                self.np.dot(fullb[loc_sl],fulla[:,loc_sl])
            with self.assertRaises(ValueError):
                self.np.dot(fulla[:,loc_sl].T, fullb[loc_sl])

    def test_dot_matrix_matrix(self):
        np.random.seed(1)

        m = 5
        n = 10
        fulla = np.random.random((m, n))
        fullb = np.random.random((n, m))

        step = n // self.MPIsize

        if self.rank == self.MPIsize - 1:
            loc_sl = slice(self.rank * step, None)
        else:
            loc_sl = slice(self.rank * step, (self.rank + 1) * step)

        np.testing.assert_allclose(self.np.dot(fulla[:, loc_sl], fullb[loc_sl,:]), np.dot(fulla, fullb))

    def test_any_scalar(self):
        rank = self.comm.Get_rank()

        locval = False

        if rank == 0:
            locval=True

        assert self.np.any(locval)

        locval = False

        assert not self.np.any(locval)

    def test_any_array(self):
        rank = self.comm.Get_rank()

        locval = np.array([False, False, True])

        if rank == 0:
            locval = np.array([False, True])

        assert self.np.any(locval)

        locval = np.array([False, False, False])

        assert not self.np.any(locval)


    def test_all_scalar(self):
        rank = self.comm.Get_rank()

        locval = True

        if rank == 0:
            locval=False

        assert not self.np.all(locval)

        locval = True

        assert  self.np.all(locval)

    def test_all_array(self):
        rank = self.comm.Get_rank()

        locval = np.array([True, True, True])

        if rank == 0:
            locval = np.array([False, True])

        assert not self.np.all(locval)

        locval = np.array([True, True, True])

        assert  self.np.all(locval)


suite = unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(test_ParallelNumpy)])

if __name__ in  ['__main__','builtins']:
    print("Running unittest test_ParallelNumpy")
    result = unittest.TextTestRunner().run(suite)