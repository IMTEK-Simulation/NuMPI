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

## plot
if toPlot:
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(xg, yg)
    ax.contour(X, Y, mat_fun(xg, yg))
    fig.show()

    print(ex_fun(np.array((0, 0, 0))))
    fig2, ax2 = plt.subplots()
    for z in [0]:
        for y in [0, 1, 2]:
            ax2.plot(xg, [ex_fun(np.array((x, y, z))) for x in xg])
            ax2.plot(xg, [ex_jac(np.array((x, y, z)))[0]for x in xg],'--')
    #plt.show(block = True)
    ######11
# Initial history:
x_old = np.array([2.,1.,-1.],dtype=float)
x_old.shape = (-1,1)
#x_old.shape=(-1,1)
grad_old = ex_jac(x_old)

# line search
reslinesearch = scipy.optimize.minimize_scalar(fun=lambda alpha: ex_fun(x_old - grad_old * alpha), bounds=(0, 2000),
                                      method="bounded")
assert reslinesearch.success
print("alpha {}".format(reslinesearch.x))
x = x_old - grad_old * reslinesearch.x
assert ex_fun(x) < ex_fun(x_old)
grad = ex_jac(x)
print("x first_linesearch {}".format(x))
k = 1
print("Wolfe")

print("1st: {}".format(first_wolfe_condition(ex_fun, x_old, ex_jac, -grad_old, reslinesearch.x, beta1=1e-4)))
print("2nd {}".format(second_wolfe_condition(x_old,ex_jac,-grad_old,reslinesearch.x,beta2=0.9)))


resscipy = scipy.optimize.minimize(ex_fun,x,jac = ex_jac, options=dict(gtol=1e-10, ftol=0))
print("scipy success: {}".format(resscipy.success))
print("scipy nit {}".format(resscipy.nit))
print("scipy result: {}".format(resscipy.x))

fig, axes = plt.subplots(3,1)

def plot(res, *args, **kwargs):
    for i, a in enumerate(axes):
        x = []
        for iterate, iteration in zip(res['iterates'], range(len(res['iterates']))):
            x.append(iterate.x[i])
        a.plot( range(len(x)), x, *args, **kwargs)

print(x)
print(x_old)

res = old_lbfgs(ex_fun, x, jac=ex_jac, x_old=x_old.copy(), maxcor=3, maxiter=100, gtol=1e-10, printdb=print)
assert res.success
print("nit {}".format(res.nit))
plot(res, "ob", label="old_lbfgs")

print(x)
print(x_old)

res = mpi_lbfgs(ex_fun, x, jac=ex_jac, x_old=x_old.copy(), maxcor=3, maxiter=100, gtol=1e-10, ftol=0,  printdb=print, store_iterates='iterate')
assert res.success
print("nit {}".format(res.nit))
plot(res, "+r", label="mpi_lbfgs")
axes[0].legend()
axes[-1].set_xlabel("iteration")

fig.show()
np.testing.assert_almost_equal(res.x, np.zeros(res.x.shape), decimal=5)

