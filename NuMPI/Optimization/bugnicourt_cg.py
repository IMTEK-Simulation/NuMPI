"""For implementing a generic form of Constrained Conjugate Gradient for mainly
 use in calculating the optimized displacement when working with contact
 mechanics problem with adhesion. This implementation is mainly based upon
 Bugnicourt et. al. - 2018, OUT_LIN algorithm.
"""

import numpy as np
import scipy.optimize as optim
from inspect import signature
from NuMPI.Tools import Reduction


def constrained_conjugate_gradients(fun, hessp,
                                    x0=None, mean_val=None,
                                    gtol=1e-8,
                                    maxiter=3000,
                                    callback=None,
                                    communicator=None,
                                    ):
    if communicator is None:
        comm = np
        nb_DOF = x0.size
    else:
        comm = Reduction(communicator)
        nb_DOF = comm.sum(x0.size)

    x = x0.copy()
    x = x.flatten()

    '''Initial Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g'''
    residual = fun(x)[1]

    mask_neg = x <= 0
    x[mask_neg] = 0.0

    if mean_val is not None:
        #
        mask_nonzero = x > 0
        N_mask_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
        residual = residual \
            - comm.sum(residual[mask_nonzero]) / N_mask_nonzero

    '''Apply the admissible Lagrange multipliers.'''
    mask_res = residual >= 0
    mask_bounded = np.logical_and(mask_neg, mask_res)
    residual[mask_bounded] = 0.0

    if mean_val is not None:
        #
        mask_nonzero = residual != 0
        N_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
        residual[mask_nonzero] = residual[mask_nonzero] - comm.sum(
            residual[mask_nonzero]) / N_nonzero
    '''INITIAL DESCENT DIRECTION'''
    des_dir = -residual

    n_iterations = 1

    for i in range(1, maxiter + 1):
        sig = signature(hessp)
        if len(sig.parameters) == 2:
            hessp_val = hessp(x, des_dir)
        elif len(sig.parameters) == 1:
            hessp_val = hessp(des_dir)
        else:
            raise ValueError('hessp function has to take max 1 arg (descent '
                             'dir) or 2 args (x, descent direction)')
        denominator_temp = comm.sum(des_dir.T * hessp_val)
        # TODO: here we could spare one FFT

        if denominator_temp == 0:
            print("it {}: denominator for alpha is 0".format(i))

        alpha = -comm.sum(residual.T * des_dir) / denominator_temp

        if alpha < 0:
            print("it {} : hessian is negative along the descent direction. "
                  "You will probably need linesearch "
                  "or trust region".format(i))

        x += alpha * des_dir

        '''finding new contact points and making the new_gap admissible
        according to these contact points.'''

        mask_neg = x <= 0
        x[mask_neg] = 0.0

        if mean_val is not None:
            x = (mean_val / comm.sum(x) * nb_DOF) * x

        residual_old = residual

        '''Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g'''
        residual = fun(x)[1]

        if mean_val is not None:
            #
            mask_nonzero = x > 0
            N_mask_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
            residual = residual \
                - comm.sum(residual[mask_nonzero]) / N_mask_nonzero

        '''Apply the admissible Lagrange multipliers.'''
        mask_res = residual >= 0
        mask_bounded = np.logical_and(mask_neg, mask_res)
        residual[mask_bounded] = 0.0

        if mean_val is not None:
            #
            mask_nonzero = residual != 0
            N_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
            residual[mask_nonzero] = residual[mask_nonzero] - comm.sum(
                residual[mask_nonzero]) / N_nonzero
            # assert np.mean(residual) < 1e-14 * np.max(abs(residual))

        '''Computing beta for updating descent direction
            beta = num / denom
            num = new_residual_transpose . (new_residual - old_residual)
            denom = alpha * descent_dir_transpose . (A_inverse - d2_ψadh).
            descent_dir '''

        # beta = np.sum(residual.T * hessp_val) / denominator_temp
        beta = comm.sum(residual * (residual - residual_old)) / (
                alpha * denominator_temp)

        des_dir_old = des_dir
        des_dir = -residual + beta * des_dir_old

        # TODO: this is useless
        des_dir[np.logical_not(mask_bounded)] = -residual[
            np.logical_not(mask_bounded)] + beta * des_dir_old[
            np.logical_not(mask_bounded)]

        des_dir[mask_bounded] = 0

        if callback:
            callback(x)

        n_iterations += 1
        # TODO: this is not parallelization friendly
        # assert np.logical_not(np.isnan(x).any())
        if i >= 5:
            if comm.max(abs(residual)) <= gtol:
                result = optim.OptimizeResult(
                    {
                        'success': True,
                        'x': x,
                        'fun': fun,
                        'jac': residual,
                        'nit': i,
                        'success': True,
                        'message': 'CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL',
                        })
                return result

            elif i == maxiter - 1:
                result = optim.OptimizeResult(
                    {
                        'success': False,
                        'x': x,
                        'fun': fun,
                        'jac': residual,
                        'nit': i,
                        'success': False,
                        'message': 'NO CONVERGENCE: MAXITERATIONS REACHED'
                        })

                return result
