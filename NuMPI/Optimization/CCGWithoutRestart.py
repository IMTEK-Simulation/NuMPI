"""
Constrained Conjugate Gradient for moderately nonlinear problems.
This implementation is mainly based upon Bugnicourt et. al. - 2018, OUT_LIN
algorithm.
"""

import numpy as np

from inspect import signature

from ..Tools import Reduction
from .Result import OptimizeResult


def constrained_conjugate_gradients(fun, hessp, x0, args=(), mean_val=None, gtol=1e-8, maxiter=3000, callback=None,
                                    communicator=None, bounds=None):
    """
    Constrained conjugate gradient algorithm from Bugnicourt et al. [1].

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. The function should return a float
        (energy) and an ndarray (gradient). Note that energy is never used, you can
        return a dummy value.
    hessp : callable
        Function to evaluate the hessian product of the objective. Hessp should
        accept either 1 argument (descent direction) or 2 arguments (x, descent
        direction).
    x0 : ndarray
        Initial guess. ValueError is raised if "None" is provided.
    gtol : float, optional
        Convergence criterion is max(abs) and norm2 of the projected gradient <
        gtol. Default value is 1e-8.
    mean_value :  int/float, optional
        If you want to apply the mean_value constraint then provide an int/float
        value to the mean_value.
    residual_plot : bool, optional
        If set to True, generates a plot between the residual and iterations.
    maxiter : int, optional
        Maximum number of iterations after which the program will exit. Default
        value is 5000.

    Returns
    -------
    OptimizeResult  : scipy.optimize object.
        The result of the optimization. The object has the following attributes:
        - success: bool, indicates if the optimization was successful
        - x: ndarray, the optimized parameters
        - jac: ndarray, the residual which is equal to the gradient at x
        - nit: int, the number of iterations performed
        - message: str, a message describing the reason for the termination

    References
    ----------
    ..[1] Bugnicourt, Romain & Sainsot, Philippe & Dureisseix, David &
        Gauthier, Catherine & Lubrecht, Ton. (2018). FFT-Based Methods
        for Solving a Rough Adhesive Contact: Description and
        Convergence Study.
    """
    if communicator is None:
        comm = np
        nb_DOF = x0.size
    else:
        comm = Reduction(communicator)
        nb_DOF = comm.sum(x0.size)

    x = x0.copy()
    x = x.flatten()

    if bounds is None:
        bounds = np.zeros_like(x)

    mask_bounds = bounds > - np.inf
    nb_bounds = comm.sum(np.count_nonzero(mask_bounds))
    mean_bounds = comm.sum(bounds) / nb_bounds

    if mean_val is not None and nb_bounds < nb_DOF:
        raise ValueError("mean_value constrained mode not compatible "
                         "with partially bound system")
        # There are ambiguities on how to compute the mean values

    '''Initial Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g'''
    residual = fun(x, *args)[1]

    mask_neg = x <= bounds
    x[mask_neg] = bounds[mask_neg]

    if mean_val is not None:
        #
        mask_nonzero = x > bounds
        N_mask_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
        residual = residual - comm.sum(residual[mask_nonzero]) / N_mask_nonzero

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
        # Here we could evaluate this directly in Fourier space (Parseval)
        # and spare one FFT.
        # See issue #47

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

        mask_neg = x <= bounds
        x[mask_neg] = bounds[mask_neg]

        if mean_val is not None:
            # x = (mean_val / comm.sum(x) * nb_DOF) * x
            # below is just a more complicated version of this compatible with
            # more general bounds
            x = bounds + (mean_val - mean_bounds) \
                / (comm.sum(x) / nb_DOF - mean_bounds) * (x - bounds)
        residual_old = residual

        '''
        In Bugnicourt's paper
        Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g
        '''
        residual = fun(x, *args)[1]

        if mean_val is not None:
            mask_nonzero = x > bounds
            N_mask_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
            residual = residual - comm.sum(
                residual[mask_nonzero]) / N_mask_nonzero

        '''Apply the admissible Lagrange multipliers.'''
        mask_res = residual >= 0
        mask_bounded = np.logical_and(mask_neg, mask_res)
        residual[mask_bounded] = 0.0

        if mean_val is not None:
            mask_nonzero = residual != 0
            N_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
            residual[mask_nonzero] = residual[mask_nonzero] - comm.sum(
                residual[mask_nonzero]) / N_nonzero
            # assert np.mean(residual) < 1e-14 * np.max(abs(residual))

        '''Computing beta for updating descent direction
            In Bugnicourt's paper:
            beta = num / denom
            num = new_residual_transpose . (new_residual - old_residual)
            denom = alpha * descent_dir_transpose . (A_inverse - d2_ψadh).
            descent_dir '''

        # beta = np.sum(residual.T * hessp_val) / denominator_temp
        beta = comm.sum(residual * (residual - residual_old)) / (
                alpha * denominator_temp)

        des_dir_old = des_dir
        des_dir = -residual + beta * des_dir_old

        des_dir[mask_bounded] = 0

        if callback:
            callback(x)

        n_iterations += 1

        if comm.max(abs(residual)) <= gtol:
            result = OptimizeResult(
                {
                    'success': True,
                    'x': x,
                    'jac': residual,
                    'nit': i,
                    'message': 'CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL',
                })
            return result

        elif i == maxiter - 1:
            result = OptimizeResult(
                {
                    'success': False,
                    'x': x,
                    'jac': residual,
                    'nit': i,
                    'message': 'NO CONVERGENCE: MAXITERATIONS REACHED'
                })

            return result
