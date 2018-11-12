
import numpy as np
import scipy.optimize

from Tools.ParallelNumpy import ParallelNumpy
from mpi4py import MPI

def donothing(*args,**kwargs):
    pass

def steepest_descent_wolfe2(x0,f,fprime, pnp = None,**kwargs):
    """
    For first Iteration there is no history. We make a steepest descent satisfying strong Wolfe Condition
    :return:
    """

    # x_old.shape=(-1,1)
    grad0 = fprime(x0)

    # line search
    alpha, phi, phi0, derphi = scipy.optimize.linesearch.scalar_search_wolfe2(
        lambda alpha: f(x0 - grad0 * alpha),
        lambda alpha: pnp.dot(fprime(x0 - grad0 * alpha).T, -grad0),**kwargs)
    assert derphi is not None, "Line Search in first steepest descent failed"
    x = x0 - grad0 * alpha

    return x, fprime(x) , x0, grad0

def LBFGS(fun, x, args=(), jac=None, x_old=None, maxcor=5, gtol = None,g2tol=1e-10, maxiter=10000,
          maxls=20,linesearch_options={}, pnp = ParallelNumpy(MPI.COMM_WORLD),store_iterates="iterate",printdb=donothing):

    if x_old is None:
        x_old = x.copy()

        x,grad,x_old,grad_old = steepest_descent_wolfe2(x_old, fun, jac,pnp=pnp)
    else:
        grad_old = np.asarray(jac(x_old))
    iterates = list()
    k = 1

    n = x.size  # Dimension of x

    gamma = 1

    S = np.zeros((n, 0))
    Y = np.zeros((n, 0))
    R = np.zeros((0, 0))
    STgrad = np.array((1, maxcor))
    YTgrad = np.array((1, maxcor))

    grad = np.asarray(jac(x))
    grad2 = pnp.sum(grad ** 2)

    alpha = 0

    # Start loop
    printdb(k)
    while True:
        # Update Sk,Yk
        if k > maxcor:
            S = np.roll(S, -1)
            S[:, -1] = (x - x_old).flat
            Y = np.roll(Y, -1)
            Y[:, -1] = (grad - grad_old).flat

        else:
            S = np.hstack([S, x - x_old])
            Y = np.hstack([Y, grad - grad_old])

        # 2.
        grad2prev = grad2.copy()
        grad2 = pnp.sum(np.asarray(grad) ** 2)  # ok

        # check if job is done
        if ((grad2 < g2tol if g2tol is not None else True) and
                (pnp.max(np.abs(grad)) < gtol if gtol is not None else True)):
            result = scipy.optimize.OptimizeResult({'success': True,
                                                    'x': x,
                                                    'fun':phi,
                                                    'jac':grad,
                                                    'nit': k,
                                                    'iterates': iterates})

            # if iterates:
            #    result['iterates'] = iterates
            return result

        if k > maxiter:
            result = scipy.optimize.OptimizeResult({'success': False,
                                                    'x': x,
                                                    'fun':phi,
                                                    'jac':grad,
                                                    'nit': k,
                                                    'iterates': iterates})

            return result

        STgrad_prev = STgrad.copy()
        YTgrad_prev = YTgrad.copy()

        STgrad = pnp.dot(S.T, grad)
        YTgrad = pnp.dot(Y.T, grad)

        if k > maxcor:
            w = np.vstack([STgrad_prev, gamma * YTgrad_prev])
            S_now_T_grad_prev = np.roll(STgrad_prev,-1)
            S_now_T_grad_prev[-1] = - alpha * gamma * grad2prev - alpha * w.T.dot(p)
        else : # straightforward Version
            S_now_T_grad_prev = pnp.dot(S.T, grad_old)


        if k > maxcor:
            R = np.roll(R, (-1, -1), axis=(0, 1))  # mxm Matrix hold by all Processors
            R[-1, :] = 0
            STym1 = STgrad - S_now_T_grad_prev
            R[:, -1] = STym1.flat  #O(m x n)

        elif k == 1:
            R = np.triu(pnp.dot(S.T, Y))
        else:
            R = np.vstack([R, np.zeros(k - 1)])
            R = np.hstack([R, pnp.dot(S.T, Y[:, -1]).reshape(k, 1)])

        if k > maxcor:
            D = np.roll(D, (-1, -1), axis=(0, 1))
            # D[-1,-1] = np.dot(Y[:,-1],Y[:,-1])# yk-1Tyk-1 # TOOPTIMIZE
            D[-1, -1] = R[-1,-1]
        else:
            #D = np.diag(np.einsum("ik,ik -> k", S, Y))
            D = np.diag(R.diagonal())
        assert D[-1,-1] >0, "k = {}: ".format(k)  # Assumption of Theorem 2.2

        if k > maxcor:
            YTY = np.roll(YTY, (-1, -1), axis=(0, 1))
            YTY[-1, :-1] = YTY[:-1, -1] = (YTgrad[:-1] - YTgrad_prev[1:]).flat
            YTY[-1, -1] = grad2prev - grad2 + 2 * YTgrad[-1]
        else:
            YTY = pnp.dot(Y.T, Y)

        gamma = D[-1, -1] / YTY[-1, -1]  # n.b. D[-1,-1] = sk-1T yk-1 = yk-1T sk-1
        Rinv = np.linalg.inv(R)

        RiSg = Rinv.dot(STgrad)
        p = np.vstack([Rinv.T.dot(D + gamma * YTY).dot(RiSg) - gamma * Rinv.T.dot(YTgrad)
                          , - RiSg])

        Hgrad = gamma * grad + np.hstack([S, gamma * Y]).dot(p)

        printdb("Linesearch: ")
        alpha, phi, phi0, derphi = scipy.optimize.linesearch.scalar_search_wolfe2(lambda alpha: fun(x - Hgrad * alpha),
                                                                                  lambda alpha: pnp.dot(
                                                                                      jac(x - Hgrad * alpha).T, -Hgrad),
                                                                                  maxiter=maxls, **linesearch_options)
        printdb("derphi: {}".format(derphi))
        assert derphi is not None, "line-search did not converge"

        x_old[:] = x
        x = x - Hgrad * alpha

        grad_old[:] = grad
        grad = jac(x)

        if store_iterates == 'iterate':
            iterate = scipy.optimize.OptimizeResult(
                {'x': x.copy(),
                 'fun': phi,
                 'jac': grad})
            iterates.append(iterate)

        printdb("k = {}".format(k))
        k = k + 1