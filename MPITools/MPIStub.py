"""
Stub implementation of mpi4py. This is necessary to run a serial version of
MPITools and dependent projects without an MPI installation.
"""

from enum import Enum


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
        return 0

    def Reduce(sendbuf, recvbuf, op=Operations.SUM, root=0):
        if root != 0:
            raise ValueError('Root must be zero for MPI stub implementation.')
        recvbuf[...] = sendbuf

COMM_WORLD = Communicator