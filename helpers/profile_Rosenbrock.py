#
# Copyright 2020 Antoine Sanner
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
from NuMPI.Tools.Reduction import Reduction
from test.Optimization.MPI_minimization_problems import MPI_Quadratic
import time

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

if comm.rank == 0:
    print("to vizualize the profile, execute `snakeviz profile_out.{rank}`")
