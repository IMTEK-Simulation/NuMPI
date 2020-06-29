import numpy as np
from NuMPI import MPI
from NuMPI.Tools.Reduction import Reduction
from tests.Optimization.MPI_minimization_problems import MPI_Objective_Interface, \
    MPI_Quadratic
import scipy.optimize
import time
import tests.Optimization.minimization_problems as mp

from NuMPI.Optimization.MPI_LBFGS_Matrix_H import LBFGS

from NuMPI import MPI
import cProfile

def profile(filename=None, comm=MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator



def timer(fun, *args, **kwargs):
    start = time.perf_counter()
    res = fun(*args, **kwargs)
    delay = time.perf_counter() - start
    return res, delay


np.random.seed(1)
n = int(1e7)


# Objective = mp.Extended_Rosenbrock
maxcor = 10
factors = 0.1 + np.random.random(n)
startpoint = np.random.normal(size=n)

comm = MPI.COMM_WORLD


pnp = Reduction(comm)
# PObjective = MPI_Objective_Interface(Objective, nb_domain_grid_pts=n, comm=comm)
PObjective = MPI_Quadratic(nb_domain_grid_pts=n, pnp=pnp, factors=factors,
                           startpoint=startpoint)
x0 = PObjective.startpoint()

LBFGS = profile("profile_out", comm)(LBFGS)
res, t = timer(LBFGS, PObjective.f, x0, jac=PObjective.grad, maxcor=maxcor,
               maxiter=100000, gtol=(1e-5), pnp=pnp)
assert res.success
if MPI.COMM_WORLD.Get_rank() == 0: print("elapsed time: {}".format(t))

if comm.rank==0:
    print("to vizualize the profile, execute `snakeviz profile_out.{rank}`")