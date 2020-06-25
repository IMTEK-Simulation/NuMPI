
from NuMPI.Optimization import LBFGS
from tests.MPI_minimization_problems import MPI_Quadratic
import numpy as np
import scipy
from NuMPI import MPI
from NuMPI.Tools import Reduction
from memory_profiler import profile
import sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs= comm.Get_size()
pnp = Reduction(comm)
print(sys.argv)
n= int(sys.argv[1])
print(n)

fp=open("LBFGS_memory_profile_n{}_nprocs{}_rank{}.log".format(n, nprocs, rank), "w")
LBFGS = profile(stream=fp)(LBFGS)

Objective = MPI_Quadratic(n, pnp=pnp)
result = LBFGS(Objective.f,Objective.startpoint(),
                jac=Objective.grad,
                options ={"gtol":1e-6,"pnp":Objective.pnp})
fp.close()
assert result.success
print(result.nit)
