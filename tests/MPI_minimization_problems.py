#
# Copyright 2019 Lars Pastewka
#           2018 Antoine Sanner
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

from MPITools import MPI
from MPITools.Tools import ParallelNumpy

class MPI_Extended_Rosenbrock(): #TODO: This doesn't work
    """

    This is the Definition like in Moré et al. and not like in scipy

    n should be even

    :param x: 1d array
    :return:
    """
    bounds = (-4, 4)

    def __init__(self,domain_resolution,pnp = ParallelNumpy()):
        raise NotImplementedError("Need to implement communication")
        comm = pnp.comm
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution // nprocs

        if rank == nprocs - 1:
            self.subdomain_slice = slice(rank * step, None)
            self.subdomain_location = rank * step
            self.subdomain_resolution = domain_resolution - rank * step
        else:
            self.subdomain_slice  = slice(rank * step, (rank + 1) * step)
            self.subdomain_location = rank * step
            self.subdomain_resolution = step

        #helps to select the data that has odd or even index in the global array
        self._sl_odd  = slice(self.subdomain_location%2,None,2)
        self._sl_even = slice((self.subdomain_location+1)%2,None,2)
        self.pnp = pnp

    def f_grad(self,x):
        x_odd = x[self._sl_odd]
        x_even = x[self._sl_even]

        sumf2 = (self.pnp.sum(100 * (x_odd - x_even ** 2) ** 2 + (1 - x_even) ** 2)).item()

        grad = np.zeros_like(x)
        grad[self._sl_odd] = 200 * (x_odd - x_even ** 2)
        grad[self._sl_even] = -400 * x_even * (x_odd - x_even ** 2) - 2 * (1 - x_even)  # # d / dx2l-1

        return sumf2, grad

    def f(self, x):
        return self.f_grad(x)[0]

    def grad(self, x):
        return self.f_grad(x)[1]

    def startpoint(self):
        """
        standard starting point
        :param n:
        :return: array of shape (1,n)
        """
        x0 = np.zeros(self.subdomain_resolution, dtype=float)
        x0.shape = (-1, 1)

        x0[self._sl_even]= -1.2
        x0[self._sl_odd]= 1

        return x0

    @staticmethod
    def minVal(*args):
        return 0

    def xmin(self):
        """
        Location of minimum according to

        Mori, J. J., Garbow, B. S. & Hillstrom, K. E. Testing Unconstrained Optimization Software. 25 (1981).

        This function not necessarily have only one Minimum in higher dimensional Space: see e.g. 10.1162/evco.2006.14.1.119

        :param n: number of DOF
        :return: array of size n
        """

        return np.ones((self.subdomain_resolution, 1), dtype=float)

class MPI_Quadratic():
    """
    n should be even

    :param x: 1d array
    :return:
    """
    bounds = (-4, 4)

    def __init__(self,domain_resolution,pnp = ParallelNumpy(),factors = None, startpoint = None):
        comm = pnp.comm
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution // nprocs

        if rank == nprocs - 1:
            self.subdomain_slice = slice(rank * step, None)
            self.subdomain_location = rank * step
            self.subdomain_resolution = domain_resolution - rank * step
        else:
            self.subdomain_slice  = slice(rank * step, (rank + 1) * step)
            self.subdomain_location = rank * step
            self.subdomain_resolution = step

        #helps to select the data that has odd or even index in the global array
        self.pnp = pnp

        if factors is not None:
            self.factors = factors[self.subdomain_slice]
        else:
            self.factors = np.random.random(self.subdomain_resolution)+0.1

        if startpoint is not None:
            self._startpoint = startpoint[self.subdomain_slice]
        else :
            self._startpoint = np.random.normal(size= self.subdomain_resolution)

    def f_grad(self,x):
        factdotx = self.factors.reshape(x.shape) * x
        return self.pnp.sum(factdotx**2,axis = 0), 2 * factdotx

    def f(self, x):
        return self.pnp.sum(np.dot((x ** 2).flat, self.factors**2), axis=0)

    def grad(self, x):
        return 2 * self.factors.reshape(x.shape) * x

    def startpoint(self):
        """
        standard starting point
        :param n:
        :return: array of shape (1,n)
        """
        return self._startpoint

    @staticmethod
    def minVal(*args):
        return 0

    def xmin(self):
        """
        Location of minimum according to
        :param n: number of DOF
        :return: array of size n
        """
        return np.zeros(self.subdomain_resolution, dtype=float)


class MPI_Objective_Interface():
    """
    creates an interface for an objective function computed serially to appear like parallel.

    That means the gradient is not the full vector but only the components corresponding to the subdomain

    """
    def __init__(self, Objective,domain_resolution, comm=MPI.COMM_WORLD):

        # define the partition between the processors

        self.comm = comm
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution // nprocs

        if rank == nprocs - 1:
            self.subdomain_slice = slice(rank * step, None)
            self.subdomain_location = rank * step
            self.subdomain_resolution = domain_resolution - rank * step
        else:
            self.subdomain_slice = slice(rank * step, (rank + 1) * step)
            self.subdomain_location = rank * step
            self.subdomain_resolution = step

        self.counts = [step] * nprocs
        self.counts[-1]=domain_resolution - (nprocs-1) * step

        self.Objective = Objective
        self.domain_resolution = domain_resolution

    def f_grad(self,x):
        x_ = x.reshape(-1)
        fullx = np.zeros(self.domain_resolution, dtype = x.dtype )
        self.comm.Allgatherv(x_,[fullx,self.counts])
        f, fullgrad = self.Objective.f_grad(fullx)
        return f, fullgrad[self.subdomain_slice].reshape(x.shape)

    def f(self, x):
        return self.f_grad(x)[0]

    def grad(self, x):
        return self.f_grad(x)[1]

    def startpoint(self):
        return self.Objective.startpoint(self.domain_resolution)[self.subdomain_slice]

    def xmin(self):
        return self.Objective.xmin(self.domain_resolution)[self.subdomain_slice]

