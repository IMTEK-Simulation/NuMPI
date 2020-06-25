import numpy as np
from NuMPI.Optimization.LBFGS_Matrix_H import LBFGS as old_lbfgs
from NuMPI.Optimization.MPI_LBFGS_Matrix_H import LBFGS as mpi_lbfgs
import scipy.optimize
from NuMPI.Optimization.Wolfe import second_wolfe_condition,first_wolfe_condition
import matplotlib.pyplot as plt

# quadratic Potential

# quadratic
toPlot = True


def ex_fun(x):
    "x should be an np array"
    #x.shape = (-1, 1)
    return np.sum(np.dot((x ** 2).flat,np.array([1,4,9])),axis=0)

def ex_jac(x):
    return 2 * np.array((1* x[0],4* x[1],9 * x[2]))


xg, yg = np.linspace(-5, 5, 51), np.linspace(-6, 6, 51)

def mat_fun(x_g, x_):
    Z = np.zeros((xg.size, yg.size))

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[j, i] = ex_fun(np.array([xg[i], yg[j], 0]))
    return Z


fig, axes = plt.subplots(3,1)

def plot(iterates, *args, **kwargs):
    for i, a in enumerate(axes):
        x = []
        for iterate, iteration in zip(iterates, range(len(iterates))):
            x.append(iterate[i])
        a.plot( range(len(x)), x, *args, **kwargs)

x = np.array([2.,1.,-1.],dtype=float)

class decorated_fun():
    def __init__(self, fun):
        self.evaluated_xs = []
        self.fun = fun
    def __call__(self, x_):
        print(x_)
        self.evaluated_xs.append(x_.reshape(-1))
        return self.fun(x_)

dec = decorated_fun(ex_fun)
res = scipy.optimize.minimize(dec, x ,jac = ex_jac, options=dict(gtol=1e-10, ftol=0))
print("scipy success: {}".format(res.success))
print("scipy nit {}".format(res.nit))
print("scipy result: {}".format(res.x))

plot(dec.evaluated_xs, label="scipy")

#res = old_lbfgs(ex_fun, x, jac=ex_jac, maxcor=3, maxiter=100, gtol=1e-10, printdb=print)
#assert res.success
#print("nit {}".format(res.nit))
#plot(res, "ob", label="old_lbfgs")


dec = decorated_fun(ex_fun)
res = mpi_lbfgs(dec, x, jac=ex_jac, maxcor=3, maxiter=100, gtol=1e-10, ftol=0,  printdb=print)
assert res.success
print("nit {}".format(res.nit))
plot(dec.evaluated_xs, "+r", label="mpi_lbfgs")
axes[0].legend()
axes[-1].set_xlabel("function evaluations")

fig.show()
np.testing.assert_almost_equal(res.x, np.zeros(res.x.shape), decimal=5)

