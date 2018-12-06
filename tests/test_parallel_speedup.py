
import numpy as np
from mpi4py import MPI
from MPITools.Tools import  ParallelNumpy
from tests.MPI_minimization_problems import MPI_Quadratic
import  time

from MPITools.Optimization import LBFGS

def timer(fun, *args, **kwargs):
    start = time.perf_counter()
    res = fun(*args, **kwargs)
    delay = time.perf_counter() - start
    return res, delay


def test_parallel_speedup():

    msg = ""

    orsizes = np.array([4, 8, 10,20])
    orsizes = orsizes[orsizes <= MPI.COMM_WORLD.size]
    sizes = orsizes.copy()

    toPlot = MPI.COMM_WORLD.Get_rank() == 0 and True
    if toPlot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax,ax2) = plt.subplots(2,1,sharex=True)
        ax.set_xlabel("nprocs")
        ax.set_ylabel("t[1proc] / t")
        ax2.set_xlabel("nprocs")
        ax2.set_ylabel("t (s)")
        ax2.set_yscale('log')

        ax.plot(sizes,sizes,'--k',label="ideal")

    #for n in [int(1e5),int(2e5),int(1e6),int(2e6),int(1e7)]:
    for n in [int(1e6)]:
        #sizes = orsizes[orsizes > n / 1e4]

        if len(sizes) == 0 : continue
        t = np.zeros(len(sizes), dtype=float)
        res = [None] * len(sizes)

        #Objective = mp.Extended_Rosenbrock
        maxcor = 5
        factors = 0.1 + np.random.random(n)
        startpoint = np.random.normal(size = n)
        for i in range(len(sizes)):
            size = sizes[i]
            color = 0 if MPI.COMM_WORLD.rank < size else 1
            if MPI.COMM_WORLD.size == size:
                comm = MPI.COMM_WORLD
            elif size == 1:
                comm = MPI.COMM_SELF
            else:
                comm = MPI.COMM_WORLD.Split(color)

            pnp = ParallelNumpy(comm)
            #PObjective = MPI_Objective_Interface(Objective, domain_resolution=n, comm=comm)
            PObjective = MPI_Quadratic(domain_resolution=n, pnp=pnp,factors = factors, startpoint=startpoint)
            x0 = PObjective.startpoint()

            if MPI.COMM_WORLD.Get_rank() == 0: print(" Before min n = {}".format(n))

            res[i],t[i] = timer(LBFGS,PObjective.f, x0, jac=PObjective.grad, maxcor=maxcor, maxiter=100000, gtol=(1e-5),store_iterates =None, pnp=pnp)
            msg += "size {}:\n".format(size)

            assert res[i].success, "Minimization faild"
            np.testing.assert_allclose(res[i].x,PObjective.xmin())

            if MPI.COMM_WORLD.Get_rank() == 0 : print("n = {}, size = {}".format(n,size))

        if toPlot :
            ax.plot(sizes, float(sizes[0] * t[0]) / t,'-o',label = "n = {}".format(n))
            ax2.plot(sizes,t,'-o',label = "n = {}".format(n))
            fig.savefig("LBFGS_parallel_speedup.png")

    if toPlot:
        ax.legend()
        fig.savefig("LBFGS_parallel_speedup.png")


if __name__ == "__main__":
    test_parallel_speedup()