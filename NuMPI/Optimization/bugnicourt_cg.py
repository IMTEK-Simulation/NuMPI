"""For implementing a generic form of Constrained Conjugate Gradient for mainly
 use in calculating the optimized displacement when working with contact
 mechanics problem with adhesion. This implementation is mainly based upon
 Bugnicourt et. al. - 2018, OUT_LIN algorithm.
"""

import numpy as np
import scipy.optimize as optim
from inspect import signature


def constrained_conjugate_gradients(fun, hessp,
                                    disp0=None, mean_val=None,
                                    gtol=1e-8,
                                    maxiter=3000, callback=None):
    fun = fun
    gtol = gtol

    x = disp0.copy()
    x = x.flatten()

    '''Initial Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g'''
    residual = fun(x)[1]

    mask_neg = x <= 0
    x[mask_neg] = 0.0
    mask_res = residual >= 0

    mask_na = np.logical_and(mask_neg, mask_res)
    residual[mask_na] = 0.0
    if mean_val is not None:
        mask_nonzero = residual != 0
        residual[mask_nonzero] = residual[mask_nonzero] - np.mean(
            residual[mask_nonzero])
    '''INITIAL DESCENT DIRECTION'''
    des_dir = -residual

    fr_ar = []
    mask = x == 0
    ar = mask.sum() / len(x)
    fr_ar = np.append(fr_ar, ar)
    grads = np.array([])
    grads = np.append(grads, np.max(abs(residual)))
    n_iterations = 1

    for i in range(1, maxiter + 1):

        sig = signature(hessp)
        if len(sig.parameters) == 2:
            hessp_val = hessp(x, des_dir)
        elif len(sig.parameters) == 1:
            hessp_val = hessp(des_dir)
        else:
            raise ValueError('hessp function has to take max 1 arg (descent '
                             'dir)\
                                    or 2 args (x, descent direction)')
        denominator_temp = np.sum(des_dir.T * hessp_val)
        # TODO: here we could spare one FFT

        if denominator_temp == 0:
            print("denominator for alpha is 0")

        alpha = -np.sum(residual.T * des_dir) / denominator_temp

        if alpha < 0 :
            print("hessian is negative along the descent direction. You will "
                  "probably need linesearch or trust region")

        x += alpha * des_dir

        '''finding new contact points and making the new_gap admissible
        according to these contact points.'''

        mask_neg = x <= 0
        x[mask_neg] = 0.0

        if mean_val is not None:
            x = (mean_val / np.mean(x)) * x

        residual_old = residual

        '''Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g'''
        residual = fun(x)[1]

        if mean_val is not None:
            #
            mask_nonzero = x > 0
            residual = residual - np.mean(residual[mask_nonzero])

        '''Apply the admissible Lagrange multipliers.'''
        mask_res = residual >= 0

        mask_bounded = np.logical_and(mask_neg, mask_res)
        residual[mask_bounded] = 0.0

        if mean_val is not None:
            #
            mask_nonzero = residual != 0
            residual[mask_nonzero] = residual[mask_nonzero] - np.mean(
                residual[mask_nonzero])
            assert np.mean(residual) < 1e-14 * np.max(abs(residual))

        '''Computing beta for updating descent direction
            beta = num / denom
            num = new_residual_transpose . (new_residual - old_residual)
            denom = alpha * descent_dir_transpose . (A_inverse - d2_ψadh).
            descent_dir '''

        # beta = np.sum(residual.T * hessp_val) / denominator_temp
        beta = np.sum(residual * (residual - residual_old)) / (
                    alpha * denominator_temp)

        des_dir_old = des_dir
        des_dir = -residual + beta * des_dir_old
        des_dir[np.logical_not(mask_bounded)] = -residual[
            np.logical_not(mask_bounded)] + beta * des_dir_old[
            np.logical_not(mask_bounded)]
        des_dir[mask_bounded] = 0

        if callback:
            callback(x)

        mask = x == 0
        frac_ar = mask.sum() / len(x)
        fr_ar = np.append(fr_ar, frac_ar)

        grads = np.append(grads, np.max(abs(residual)))
        n_iterations += 1
        assert np.logical_not(np.isnan(x).any())

        if i >= 9:
            if np.max(abs(residual)) <= gtol:
                result = optim.OptimizeResult({'success': True,
                                               'x': x,
                                               'fun': fun,
                                               'jac': residual,
                                               'nit': i,
                                               'message': 'CONVERGENCE: '
                                                          'NORM_OF_GRADIENT_<='
                                                          '_GTOL',
                                               })

                # import matplotlib.pyplot as plt

                # fig, (er_, ar_) = plt.subplots(2, 1)
                # er_.plot(range(n_iterations), np.log10(grads),
                #          label='OUT-LIN')
                # # er_.set_xlabel('iterations')
                # er_.set_ylabel('Error')
                # ar_.plot(range(n_iterations),np.log10(fr_ar), label =
                # 'OUT-LIN')
                # ar_.set_xlabel('iterations')
                # ar_.set_ylabel('Frac Area')
                # er_.legend()
                # ar_.legend()
                # plt.style.use(
                # '~/Downloads/Thesis/code/SindhuThesis/plot_styles
                # /presentation.mplstyle')
                # plt.show(block=True)
                # plt.savefig("converged_bug.png")
                return result

            elif i == maxiter - 1:
                result = optim.OptimizeResult({'success': False,
                                               'x': x,
                                               'fun': fun,
                                               'jac': residual,
                                               'nit': i,
                                               'message': 'NO CONVERGENCE: '
                                                          'MAXITERATIONS '
                                                          'REACHED'
                                                  })

                # import matplotlib.pyplot as plt
                # plt.plot(range(n_iterations), np.log10(grads),
                #         label='residuals')
                # plt.xlabel('iterations')
                # plt.ylabel('residuals')
                # plt.show(block=True)
                # plt.savefig('max_iter_bugnicourt.png')
                return result
