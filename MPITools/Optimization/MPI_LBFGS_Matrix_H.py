#
# Copyright 2019 Lars Pastewka
#           2018-2019 Antoine Sanner
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
import scipy.optimize

from MPITools import MPI
from MPITools.Tools import ParallelNumpy

def donothing(*args,**kwargs):
    pass

def steepest_descent_wolfe2(x0,f,fprime, pnp = None,maxiter=10,**kwargs):
    """
    For first Iteration there is no history. We make a steepest descent satisfying strong Wolfe Condition
    :return:
    """

    # x_old.shape=(-1,1)
    grad0 = fprime(x0)

    def _fun(alpha):
        val = f(x0 - grad0 * alpha)
        return val

    def _fprime(alpha):
        val=np.asscalar(pnp.dot(fprime(x0 - grad0 * alpha).T, -grad0))
        return val

    alpha, phi, phi0, derphi = scipy.optimize.linesearch.scalar_search_wolfe2(
        _fun,
        _fprime, maxiter=maxiter, **kwargs)

    assert derphi is not None, "Line Search in first steepest descent failed"
    x = x0 - grad0 * alpha

    return x, fprime(x) , x0, grad0, phi,phi0

def LBFGS(fun, x, args=(), jac=None, x_old=None, maxcor=5, gtol = 1e-5,g2tol=None, ftol= None,maxiter=10000,
          maxls=20,linesearch_options={}, pnp=ParallelNumpy(MPI.COMM_WORLD),store_iterates=None,printdb=donothing,**options):
    #print("jac = {}, type = {}".format(jac, type(jac)))
    if jac is True: # TODO: Temp
        jac = lambda x: fun(x)[1]
        _fun = lambda x: fun(x)[0]
    elif jac is False:
        raise NotImplementedError
    else: # function provided
        _fun = fun

    original_shape=x.shape

    x = x.reshape((-1,1))
    _jac = lambda x: jac(x).reshape((-1,1))
    if x_old is None:
        x_old = x.copy()
        x,grad,x_old,grad_old,phi,phi_old = steepest_descent_wolfe2(x_old, _fun, _jac, pnp=pnp, maxiter=maxls)
    else:
        grad_old = np.asarray(_jac(x_old))
        phi_old = _fun(x_old)
        phi = _fun(x)
    iterates = list()
    k = 1

    n = x.size  # Dimension of x
    gamma = 1

    S = np.zeros((n, 0))
    Y = np.zeros((n, 0))
    R = np.zeros((0, 0))
    STgrad = np.array((1, maxcor))
    YTgrad = np.array((1, maxcor))

    grad = np.asarray(_jac(x))
    grad2 = pnp.sum(grad ** 2)

    alpha = 0

    # Start loop
    #printdb(k)
    while True:
        # Update Sk,Yk
        #print("k= {}".format(k))
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
                (pnp.max(np.abs(grad)) < gtol if gtol is not None else True) and
                ((phi - phi_old) / max((1,abs(phi),abs(phi_old))) <= ftol if ftol is not None else True)):
            result = scipy.optimize.OptimizeResult({'success': True,
                                                    'x': x.reshape(original_shape),
                                                    'fun':phi,
                                                    'jac':grad.reshape(original_shape),
                                                    'nit': k,
                                                    'iterates': iterates})
            # if iterates:
            #    result['iterates'] = iterates
            return result

        if k > maxiter:
            result = scipy.optimize.OptimizeResult({'success': False,
                                                    'x': x.reshape(original_shape),
                                                    'fun':phi,
                                                    'jac':grad.reshape(original_shape),
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

        phi_old = float(phi)
        #printdb("Linesearch: ")
        alpha, phi, phi0, derphi = scipy.optimize.linesearch.scalar_search_wolfe2(lambda alpha: _fun(x - Hgrad * alpha),
                                                                                  lambda alpha: pnp.dot(
                                                                                      _jac(x - Hgrad * alpha).T, -Hgrad),
                                                                                  maxiter=maxls, **linesearch_options)
        printdb("derphi: {}".format(derphi))
        assert derphi is not None, "line-search did not converge"

        x_old[:] = x
        x = x - Hgrad * alpha

        grad_old[:] = grad
        grad = _jac(x)

        if store_iterates == 'iterate':
            iterate = scipy.optimize.OptimizeResult(
                {'x': x.copy().reshape(original_shape),
                 'fun': phi,
                 'jac': grad.reshape(original_shape)})
            iterates.append(iterate)

        printdb("k = {}".format(k))
        k = k + 1