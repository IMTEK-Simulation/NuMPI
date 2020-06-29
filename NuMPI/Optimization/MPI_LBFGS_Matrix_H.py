#
# Copyright 2018, 2020 Antoine Sanner
#           2019 Lars Pastewka
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

# skip undefined names
# flake8: noqa F821


import numpy as np
import scipy.optimize

from NuMPI import MPI
from NuMPI.Tools import Reduction

from NuMPI.Optimization.linesearch import scalar_search_wolfe2

def donothing(*args, **kwargs):
    pass


def steepest_descent_wolfe2(x0, f_gradf, pnp=None, maxiter=10, **kwargs):
    """
    For first Iteration there is no history. We make a steepest descent
    satisfying strong Wolfe Condition
    :return:
    """
    # x_old.shape=(-1,1)
    phi0, grad0 = f_gradf(x0)
    grad0 = grad0.reshape((-1, 1))

    derphi0 = pnp.dot(grad0.T, -grad0).item()
    gradf = np.zeros_like(grad0)

    # stores the last evaluated gradf,
    # take care, the linesearch has to return the last evaluated value
    # TODO: Dangerous
    def _phi_phiprime(alpha):
        phi, gradf[...] = f_gradf(x0 - grad0 * alpha)
        phiprime = pnp.dot(gradf.T, -grad0).item()
        return phi, phiprime

    # like in lbfgsb
    # if (iter.eq. 0.and..not.boxed) then
    #    stp = min(one / dnorm, stpmx)

    alpha, phi, phi0, derphi = scalar_search_wolfe2(_phi_phiprime,
                                                    maxiter=maxiter,
                                                    phi0=phi0, derphi0=derphi0,
                                                    step=min(1., 1. / np.sqrt(
                                                        -derphi0)), **kwargs)

    assert derphi is not None, "Line Search in first steepest descent failed"
    x = x0 - grad0 * alpha

    return x, gradf, x0, grad0, phi, phi0, derphi


def LBFGS(fun, x, args=(), jac=None, x_old=None, maxcor=10, gtol=1e-5,
          ftol=2.2e-9, maxiter=15000,
          maxls=20, linesearch_options=dict(c1=1e-3, c2=0.9),
          pnp=Reduction(MPI.COMM_WORLD),
          store_iterates=None, printdb=donothing, callback=None, **options):
    r"""

    convergence if |grad|_{\infty} <= gtol or <= ftol is satisfied

    Parameters
    ----------
    fun
    x
    args
    jac
    x_old: initial guess
    maxcor: max number of history gradients stored
    gtol:
    ftol:
    maxiter
    maxls, default 20, as in scipy.optimize.fmin_l_bfgs_b
    linesearch_options: further options for the linesearch
    the result of the linesearch has to satisfy the strong wolfe condition
    See Wright and Nocedal, 'Numerical Optimization',p.34
    c1 parameter for the sufficient decrease condition
    c2 parameter for the curvature condition
    
    default values are choosen here to match the implementation in
    the Fortran subroutine L-BFGS-B 3.0 by
    Ciyou Zhu, Richard Byrd, and Jorge Nocedal

    See lbfgsb.f line 2497, with gtol=c2 and ftol=c1
    
    pnp
    store_iterates: stores each iterate of x, only debugging
    printdb
    options

    Returns
    -------

    """

    if callback is None:
        callback = donothing

    # print("jac = {}, type = {}".format(jac, type(jac)))
    if jac is True:
        def fun_grad(x):
            """
            reshapes the gradient in a convenient form
            Parameters
            ----------
            x

            Returns
            -------

            """
            f, grad = fun(x)
            return f, grad.reshape((-1, 1))

    elif jac is False:
        raise NotImplementedError(
            "Numerical evaluation of gradient not implemented")
    else:
        # function and gradient provided sepatately

        def fun_grad(x):
            """
            evaluates function and grad consequently (important, see issue #13)
            and reshapes the gradient in a convenient form
            Parameters
            ----------
            x

            Returns
            -------

            """
            # order matte
            f = fun(x)
            grad = jac(x).reshape((-1, 1))
            return f, grad

    # user can provide x in the shape of his convenience
    # but we will work with the shape (-1, 1)
    original_shape = x.shape

    x = x.reshape((-1, 1))

    if x_old is None:
        x_old = x.copy()
        x, grad, x_old, grad_old, phi, phi_old, derphi = \
            steepest_descent_wolfe2(x_old, fun_grad, pnp=pnp, maxiter=maxls,
                                    **linesearch_options)
    else:
        phi_old, grad_old = fun_grad(
            x_old)  # phi_old is never used, except for the convergence
        # criterion
        phi, grad = fun_grad(x)

    # full history of x is sored here if wished
    iterates = list()
    k = 1

    # n = x.size  # number of degrees of freedom
    gamma = 1

    Slist = []  # history of the steps of x
    Ylist = []  # history of gradient differences
    R = np.zeros((0, 0))

    grad2 = pnp.sum(grad ** 2)

    alpha = 0  # line search step size

    # Start loop
    # printdb(k)
    while True:
        # Update Sk,Yk
        # print("k= {}".format(k))
        if k > maxcor:
            # S = np.roll(S, -1)
            # S[:, -1] = (x - x_old).flat

            Slist[:-1] = Slist[1:]
            Slist[-1] = (x - x_old)

            # Y = np.roll(Y, -1)
            # Y[:, -1] = (grad - grad_old).flat

            Ylist[:-1] = Ylist[1:]
            Ylist[-1] = (grad - grad_old)

        else:
            # S = np.hstack([S, x - x_old])
            Slist.append(x - x_old)
            # Y = np.hstack([Y, grad - grad_old])
            Ylist.append(grad - grad_old)

        # 2.
        grad2prev = grad2
        grad2 = pnp.sum(np.asarray(grad) ** 2)

        ######################
        # check if job is done
        # if ((grad2 < g2tol if g2tol is not None else True) and
        #        (pnp.max(np.abs(grad)) < gtol if gtol is not None else
        #        True) and
        #        ((phi - phi_old) / max((1,abs(phi),abs(phi_old))) <= ftol
        #        if ftol is not None else True)):
        callback(x)

        if (pnp.max(np.abs(grad)) < gtol):
            result = scipy.optimize.OptimizeResult({
                'success': True,
                'x': x.reshape(
                    original_shape),
                'fun': phi,
                'jac': grad.reshape(
                    original_shape),
                'nit': k,
                'message':
                    'CONVERGENCE: '
                    'NORM_OF_GRADIENT_<=_GTOL',
                'iterates': iterates
            })
            # if iterates:
            #    result['iterates'] = iterates
            return result

        if ((phi_old - phi) <= ftol * max((1, abs(phi), abs(phi_old)))):
            result = scipy.optimize.OptimizeResult({
                'success': True,
                'x': x.reshape(
                    original_shape),
                'fun': phi,
                'jac': grad.reshape(
                    original_shape),
                'nit': k,
                'message':
                    'CONVERGENCE: '
                    'REL_REDUCTION_OF_F_<=_FACTR*EPSMCH',
                'iterates': iterates
            })
            # if iterates:
            #    result['iterates'] = iterates
            return result

        if k > maxiter:
            result = scipy.optimize.OptimizeResult({
                'success': False,
                'x': x.reshape(
                    original_shape),
                'fun': phi,
                'jac': grad.reshape(
                    original_shape),
                'nit': k,
                'iterates': iterates
            })

            return result
        ###########
        # new iteration

        STgrad = np.array([pnp.dot(si.T, grad) for si in Slist]).reshape(-1, 1)
        YTgrad = np.array([pnp.dot(yi.T, grad) for yi in Ylist]).reshape(-1, 1)
        # STgrad = pnp.dot(S.T, grad)
        # YTgrad = pnp.dot(Y.T, grad)

        if k > maxcor:
            S_now_T_grad_prev = np.roll(STgrad_prev, -1)
            S_now_T_grad_prev[-1] = - alpha * gamma * grad2prev - alpha * (
                    STgrad_prev.T.dot(p1) + gamma * YTgrad_prev.T.dot(p2))
        else:  # straightforward Version
            S_now_T_grad_prev = np.array(
                [pnp.dot(si.T, grad_old) for si in Slist]).reshape(-1, 1)

        if k > maxcor:
            R = np.roll(R, (-1, -1),
                        axis=(0, 1))  # mxm Matrix hold by all Processors
            R[-1, :] = 0
            STym1 = STgrad - S_now_T_grad_prev
            R[:, -1] = STym1.flat  # O(m x n)

        elif k == 1:
            # Rm = np.triu(pnp.dot(S.T, Y))
            R = np.triu(np.array(
                [[pnp.dot(si.T, yi).item() for yi in Ylist] for si in Slist]))
            # print(R)
        else:
            # Rm = np.vstack([Rm, np.zeros(k - 1)])
            # Rm = np.hstack([Rm, pnp.dot(S.T, Y[:, -1]).reshape(k, 1)])
            R = np.vstack([R, np.zeros(k - 1)])
            R = np.hstack([R, np.array(
                [pnp.dot(si.T, Ylist[-1]) for si in Slist]).reshape(k, 1)])
        if k > maxcor:
            D = np.roll(D, (-1, -1), axis=(0, 1))
            # D[-1,-1] = np.dot(Y[:,-1],Y[:,-1])# yk-1Tyk-1 # TOOPTIMIZE
            D[-1, -1] = R[-1, -1]
        else:
            # D = np.diag(np.einsum("ik,ik -> k", S, Y))
            D = np.diag(R.diagonal())
        assert D[-1, -1] > 0, "k = {}: ".format(k)  # Assumption of Theorem 2.2

        if k > maxcor:
            YTY = np.roll(YTY, (-1, -1), axis=(0, 1))
            YTY[-1, :-1] = YTY[:-1, -1] = (YTgrad[:-1] - YTgrad_prev[1:]).flat
            YTY[-1, -1] = grad2prev - grad2 + 2 * YTgrad[-1]
        else:
            # YTYm = pnp.dot(Y.T, Y)
            YTY = np.array(
                [[pnp.dot(yi1.T, yi2).item() for yi2 in Ylist] for yi1 in
                 Ylist])
        # Step 5.
        gamma = D[-1, -1] / YTY[
            -1, -1]  # n.b. D[-1,-1] = sk-1T yk-1 = yk-1T sk-1

        # Step 6. and 7. together
        Rinv = np.linalg.inv(R)
        RiSg = Rinv.dot(STgrad)
        p1 = Rinv.T.dot(D + gamma * YTY).dot(RiSg) - gamma * Rinv.T.dot(YTgrad)
        p2 = - RiSg  # TODO

        # temphstack=np.hstack([S, gamma * Y])

        # Hgradm = gamma * grad + S.dot(p1)  + gamma * Y.dot(p2)
        Hgrad = gamma * grad
        for si, yi, p1i, p2i in zip(Slist, Ylist, p1.flat, p2.flat):
            Hgrad += si * p1i.item() + gamma * yi * p2i.item()

        phi_old = float(phi)
        # printdb("Linesearch: ")
        grad_old[:] = grad
        x_old[:] = x

        def _phi_phiprime(alpha):
            phi, grad[...] = fun_grad(x - Hgrad * alpha)
            phiprime = pnp.dot(grad.T, -Hgrad).item()
            return phi, phiprime

        # TODO: oldphi0: is it allowed to stay outside of the search
        #  direction ?
        alpha, phi, phi0, derphi = scalar_search_wolfe2(
            _phi_phiprime,
            phi0=phi,
            derphi0=pnp.dot(grad.T, -Hgrad).item(),
            maxiter=maxls,
            **linesearch_options)

        printdb("derphi: {}".format(derphi))
        assert derphi is not None, "line-search did not converge"

        x = x - Hgrad * alpha

        if store_iterates == 'iterate':
            iterate = scipy.optimize.OptimizeResult(
                {
                    'x': x.copy().reshape(original_shape),
                    'fun': phi,
                    'jac': grad.reshape(original_shape)
                })
            iterates.append(iterate)

        printdb("k = {}".format(k))
        k = k + 1

        STgrad_prev = STgrad.copy()
        YTgrad_prev = YTgrad.copy()
