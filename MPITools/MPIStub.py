"""
Stub implementation of mpi4py. This is necessary to run a serial version of
MPITools and dependent projects without an MPI installation.
"""

from enum import Enum

import numpy as np


### Data types

class Typedict(object):
    def __getitem__(self, item):
        return np.dtype(item)

_typedict = Typedict()


### Operations

class Operations(Enum):
    MIN = 1
    MAX = 2
    SUM = 3
    PROD = 4
    LAND = 5
    BAND = 6
    LOR = 7
    BOR = 8
    LXOR = 9
    BXOR = 10
    MAXLOC = 11
    MINLOC = 12

MIN = Operations.MIN
MAX = Operations.MAX
SUM = Operations.SUM
PROD = Operations.PROD
LAND = Operations.LAND
BAND = Operations.BAND
LOR = Operations.LOR
BOR = Operations.BOR
LXOR = Operations.LXOR
BXOR = Operations.BXOR
MAXLOC = Operations.MAXLOC
MINLOC = Operations.MINLOC

### Stub communicator object

class Communicator(object):
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Reduce(self, sendbuf, recvbuf, op=Operations.SUM, root=0):
        if root != 0:
            raise ValueError('Root must be zero for MPI stub implementation.')

        try:
            senddata, sendtype = sendbuf
        except:
            senddata = sendbuf
            sendtype = sendbuf.dtype

        try:
            recvdata, recvtype = recvbuf
        except:
            recvdata = recvbuf
            recvtype = recvbuf.dtype

        if sendtype != recvtype:
            raise TypeError('Mismatch in send and receive MPI datatypes in MPI stub implementation.')

        recvdata[...] = senddata
    Allreduce = Reduce

COMM_WORLD = Communicator()