#
# Copyright 2019 Lars Pastewka
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

def donothing(*args,**kwargs):
    pass

def getLogfile():
    def savelog(string):
        with open("LBFGS.log", "w") as file:
            file.write("{}".format(string))
    return savelog
    raise BlockingIOError

def steepest_descent_wolfe2(x0,f,fprime,**kwargs):
    """
    For first Iteration there is no history. We make a steepest descent satisfying strong Wolfe Condition
    :return:
    """

    # x_old.shape=(-1,1)
    grad0 = fprime(x0)

    # line search
    alpha, phi, phi0, derphi = scipy.optimize.linesearch.scalar_search_wolfe2(
        lambda alpha: f(x0 - grad0 * alpha),
        lambda alpha: np.dot(fprime(x0 - grad0 * alpha).T, -grad0),**kwargs)
    assert derphi is not None, "Line Search in first steepest descent failed"
    x = x0 - grad0 * alpha

    return x, fprime(x) , x0, grad0



def LBFGS(fun, x, args=(), jac=None, x_old=None, maxcor=5, gtol = None, g2tol=1e-10, maxiter=10000,
          maxls=20, store_iterates="iterate", printdb=donothing, linesearch_options={}):
    """
        This implementation follows

    Byrd, R. H., Nocedal, J. & Schnabel, R. B. REPRESENTATIONS OF QUASI-NEWTON MATRICES AND THEIR USE IN LIMITED MEMORY METHODS. 31

    The signature of the function is such that you can give her as Method in scipy.optimize.minimize

    :param args:
    :param ftol:
    :param maxls:
    :param fun:
    :param jac:
    :param x:
    :param x_old:
    :param maxcor:
    :param g2tol:
    :param maxiter:
    :param store_iterates:
    :param printdb:
    :return:
    """



    if x_old is None:
        x_old = x.copy()

        x,grad,x_old,grad_old = steepest_descent_wolfe2(x_old, fun, jac,**linesearch_options)

    k=1

    n = x.size # Dimension of x

    gamma = 1

    S = np.zeros((n, 0))
    Y = np.zeros((n, 0))
    R = np.zeros((0, 0))
    STgrad = np.array((1, maxcor))
    YTgrad = np.array((1, maxcor))

    grad = np.asarray(jac(x))
    grad2 = np.sum(grad**2)
    grad_old = np.asarray(jac(x_old))

    alpha=0

    # Start loop

    iterates = list()
    if store_iterates == 'iterate':
        iterate = scipy.optimize.OptimizeResult(
            {'x': x_old.copy(),
             'fun': fun(x_old),
             'jac': grad_old.copy()})
        iterates.append(iterate)

        iterate = scipy.optimize.OptimizeResult(
            {'x': x.copy(),
             'fun': fun(x),
             'jac': grad})
        iterates.append(iterate)


    while True:
        #printdb(k)
        #printdb("grads")
        #printdb(grad)
        #printdb(grad_old)

        # Update Sk,Yk
        if k > maxcor:
            S = np.roll(S, -1)
            S[:, -1] = (x - x_old).flat
            Y = np.roll(Y, -1)
            Y[:, -1] = (grad - grad_old).flat

        else:
            S = np.hstack([S, x - x_old])
            Y = np.hstack([Y, grad - grad_old])
        #printdb("S: {}".format(S))
        #printdb("Y: {}".format(Y))

        # 2.
        grad2prev = grad2.copy()
        grad2 = np.sum(grad ** 2)  # ok

        # check if job is done
        if  ((grad2 < g2tol if g2tol is not None else True) and
            (np.max(np.abs(grad)) < gtol if gtol is not None else True)):
            result = scipy.optimize.OptimizeResult({'success': True,
                                                    'x': x,
                                                    'nit': k,
                                                    'iterates': iterates})

            #if iterates:
            #    result['iterates'] = iterates
            return result

        if k > maxiter:
            result = scipy.optimize.OptimizeResult({'success': False,
                                                    'x': x,
                                                    'nit': k,
                                                    'iterates':iterates})

            #if iterates:
            #    result['iterates'] = iterates
            return result

        STgrad_prev = STgrad.copy()
        YTgrad_prev = YTgrad.copy()

        STgrad = np.dot(S.T, grad)
        YTgrad = np.dot(Y.T, grad)  # OK, this is new

        #printdb("STgrad : {}".format(STgrad))
        #printdb("YTgrad: {}".format(YTgrad))

        if k > maxcor:
            w = np.vstack([STgrad_prev, gamma * YTgrad_prev])
            S_now_T_grad_prev = np.roll(STgrad_prev,-1)
            S_now_T_grad_prev[-1] = - alpha * gamma * grad2prev - alpha * w.T.dot(p)
        else : # straightforward Version
            S_now_T_grad_prev = np.dot(S.T, grad_old)

        #printdb("S_now_T_grad_prev {}".format(S_now_T_grad_prev))
        #np.testing.assert_allclose(S_now_T_grad_prev,np.dot(S.T, grad_old),
        #                           err_msg="Maybe the assumption of Theorem 2.2"
        #                                "is not valid: sk-1Tyk-1 = {}".format(S[:,-1].T.dot(Y[:,-1])))

        # 3. # TOOPTIMIZE
        #sprevTgradprev = np.dot(S[:, -1].T, grad_old)  # sk-1T gk-1

        #%% 4.
        #ykm12 = np.dot(Y[:, -1].T, Y[:, -1]) #TOOPTIMIZE

        #printdb("before")
        #printdb("R: {}".format(R))
        if k > maxcor:
            R = np.roll(R, (-1, -1), axis=(0, 1))  # mxm Matrix hold by all Processors
            R[-1, :] = 0
            STym1 = STgrad - S_now_T_grad_prev
            R[:, -1] = STym1.flat  #O(m x n)

        elif k == 1:
            R = np.triu(np.dot(S.T, Y))
        else:
            R = np.vstack([R, np.zeros(k - 1)])
            R = np.hstack([R, np.dot(S.T, Y[:, -1]).reshape(k, 1)])

        #np.testing.assert_allclose(R, np.triu(np.dot(S.T, Y)))

        if k > maxcor:
            D = np.roll(D, (-1, -1), axis=(0, 1))
            # D[-1,-1] = np.dot(Y[:,-1],Y[:,-1])# yk-1Tyk-1 # TOOPTIMIZE
            D[-1, -1] = R[-1,-1]
        else:
            #D = np.diag(np.einsum("ik,ik -> k", S, Y))
            D=np.diag(R.diagonal())

        assert D[-1,-1] >0, "k = {}: ".format(k)  # Assumption of Theorem 2.2
        #np.testing.assert_allclose(np.diag(D), np.diag(R))

        # YTY = np.dot(Y.T,Y) #TOPTIMIZED
        if k > maxcor:
            YTY = np.roll(YTY, (-1, -1), axis=(0, 1))
            #printdb(YTgrad)
            #printdb(YTgrad_prev)
            YTY[-1, :-1] = YTY[:-1, -1] = (YTgrad[:-1] - YTgrad_prev[1:]).flat
            YTY[-1, -1] = grad2prev - grad2 + 2 * YTgrad[-1]
        else:
            YTY = np.dot(Y.T, Y)
        #np.testing.assert_allclose(YTY, np.dot(Y.T, Y))
        ##
        #printdb("after")
        #printdb("R: {}".format(R))
        #printdb("YTY: {}".format(YTY))
        #printdb("D: {}".format(D))

        #%% 5.
        gamma = D[-1, -1] / YTY[-1,-1]  # n.b. D[-1,-1] = sk-1T yk-1 = yk-1T sk-1

        #%% 6.
        #Rinv = np.linalg.inv(R)
        Rinv = scipy.linalg.solve_triangular(R,np.eye(min(k,maxcor)))

        RiSg = Rinv.dot(STgrad)

        p = np.vstack([Rinv.T.dot(D + gamma * YTY).dot(RiSg) - gamma * Rinv.T.dot(YTgrad)
                          , - RiSg])

        #%% 7.
        Hgrad = gamma * grad + np.hstack([S, gamma * Y]).dot(p)

        #%% linesearch

        #reslinesearch = scipy.optimize.minimize_scalar(fun=lambda alpha: fun(x - Hgrad * alpha), bounds=(0, 10), method="bounded")
        #assert reslinesearch.success, "Linesearch not converged"
        # line_search did cause problems, maybe because of the differentz interpretation of the arrays
        #alpha,fc,gc,new_fval,old_fval,new_slope = scipy.optimize.line_search(fun,lambda x_ : fprime(x_).flatten(),x, - Hgrad.flatten() , c1=1e-4,c2=0.9,maxiter=20)

        #printdb("assert descent direction")
        #assert fun(x - Hgrad * 0.001) - fun(x) < 0
        #printdb(fun(x - Hgrad * 0.001) - fun(x))
        
        alpha,phi,phi0,derphi = scipy.optimize.linesearch.scalar_search_wolfe2(lambda alpha: fun(x - Hgrad * alpha), lambda alpha: np.dot(jac(x - Hgrad * alpha).T, -Hgrad),maxiter = maxls,**linesearch_options)

        if derphi is None:
            import matplotlib.pyplot as plt
            figdebug,axdebug=plt.subplots()
            alphas = np.linspace(-1,10)

            axdebug.plot(alphas,[fun(x - a * Hgrad) for a in alphas] )
            figdebug.show()
            printdb("scalar line search did not converge")
            printdb("alpha: {}".format(alpha))
            plt.show(block=True)

        assert derphi is not None, "scalar line-search did not converge"
        #assert new_fval is not None, "Line-search didn't converge"
        #printdb("x: {}".format(x))
        x_old[:] = x
        #printdb("x_old: {}".format(x_old))
        x = x - Hgrad * alpha
        #printdb("x: {}".format(x))
        #printdb("x_old: {}".format(x_old))
        #printdb("x = {}".format(x))
        #assert phi < phi0, "f(x) >= f(x_old) ! "
        grad_old[:] = grad
        grad = jac(x)
        #assert fun(x) <= fun(x_old) + 1e-4 * alpha * grad_old.T.dot(-Hgrad), "First Wolfe Condition not fullfilled"
        #assert grad.T.dot(-Hgrad) >= 0.9 * grad_old.T.dot(-Hgrad), "second Wolfe Condition not fullfilled"
        #printdb("dx * -Hgrad:{}".format((x-x_old).T.dot(-Hgrad)))
        #printdb(alpha)
        #assert (grad - grad_old).T.dot(x - x_old) > 0, "msg derphi = {}".format(derphi)
        if store_iterates == 'iterate':
            iterate = scipy.optimize.OptimizeResult(
                {'x': x.copy(),
                 'fun': phi,
                 'jac': grad})
            iterates.append(iterate)

        k = k + 1

#alpha = np.linspace(0, 2)
#plt.plot(alpha, [test_fun(x_old - grad_old * a) for a in alpha])
#plt.axvline(step)