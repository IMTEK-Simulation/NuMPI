import numpy as np

from inspect import signature

from .Result import OptimizeResult


def constrained_conjugate_gradients(fun, hessp, x0, args=(), gtol=1e-8,
                                    mean_value=None, residual_plot=False,
                                    maxiter=5000):
    """
    Implementation of constrained conjugate gradient algorithm as described in,
    I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999).

    This function minimizes a given objective function using the constrained
    conjugate gradient algorithm. The algorithm is described in detail in the
    references provided.

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
    mean_value :  float, optional
        If you want to apply the mean_value constraint then provide a float value
        to the mean_value.
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
    ..[1] I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)
    """
    if not isinstance(mean_value, (type(None), int, float)):
        raise ValueError('Inappropriate type: {} for mean_value whereas a '
                         'float \
            or int is expected'.format(type(mean_value)))

    if not isinstance(residual_plot, bool):
        raise ValueError('Inappropriate type: {} for "residual_plot" whereas '
                         'a bool \
                         is expected'.format(type(residual_plot)))

    if x0 is not None:
        x = x0.copy()
        x = x.flatten()
        delta = 0
        G_old = 1
    else:
        raise ValueError('Input required for x0/initial value !!')

    gaps = np.array([])
    iterations = np.array([])

    des_dir = np.zeros(np.shape(x))

    if residual_plot:
        gaps = np.append(gaps, 0)
        iterations = np.append(iterations, 0)

    n_iterations = 1

    while (n_iterations < maxiter + 1):

        '''Mask to truncate the negative values'''
        mask_neg = x <= 0
        x[mask_neg] = 0.0

        '''Initial residual or GAP'''
        residual = fun(x, *args)[1]

        mask_c = x > 0
        if mean_value is not None:
            residual = residual - np.mean(residual[mask_c])

        G = np.sum(residual[mask_c] ** 2)

        des_dir[mask_c] = -residual[mask_c] + delta * (G / G_old) * des_dir[
            mask_c]
        des_dir[np.logical_not(mask_c)] = 0
        G_old = G

        '''Calculating step-length alpha'''

        sig = signature(hessp)
        if len(sig.parameters) == 2:
            hessp_val = hessp(x, des_dir)
        elif len(sig.parameters) == 1:
            hessp_val = hessp(des_dir)
        else:
            raise ValueError('hessp function has to take max 1 arg (descent '
                             'dir) or 2 args (x, descent direction)')

        '''Here hessp_val is used as r_ij in original algorithm'''
        if mean_value is not None:
            hessp_val = hessp_val - np.mean(hessp_val[mask_c])

        if mask_c.sum() != 0:
            '''alpha is TAU from algorithm'''
            alpha = -np.sum(residual[mask_c] * des_dir[mask_c]) / np.sum(hessp_val[mask_c] * des_dir[mask_c])
        else:
            # TODO: does anything happen when alpha is 0 or is the algorithm just stuck ?
            alpha = 0.0

        if alpha < 0:
            print("it {} : hessian is negative along the descent direction. "
                  "You will probably need linesearch "
                  "or trust region".format(n_iterations))

        x[mask_c] += alpha * des_dir[mask_c]

        '''mask for contact'''
        mask_neg = x <= 0
        '''truncating negative values'''
        x[mask_neg] = 0.0

        mask_g = residual < 0
        mask_overlap = np.logical_and(mask_neg, mask_g)

        if mask_overlap.sum() == 0:
            delta = 1
        else:
            delta = 0
            x[mask_overlap] = x[mask_overlap] - alpha * residual[mask_overlap]

        if mean_value is not None:
            '''Take care of constraint a_x*a_y*sum(p_ij)=P0'''
            P = np.mean(x)
            x *= (mean_value / P)

        if residual_plot:
            iterations = np.append(iterations, n_iterations)
            if mask_c.sum() != 0:
                gaps = np.append(gaps, np.max(abs(residual[mask_c])))
            else:
                gaps = np.append(gaps, np.max(abs(residual)))

        n_iterations += 1
        res_convg = False
        assert np.logical_not(np.isnan(x).any())

        if n_iterations >= 3:
            '''If converged'''
            if mask_c.sum() != 0:
                if np.max(abs(residual[mask_c])) <= gtol:
                    res_convg = True
                else:
                    res_convg = False

            if res_convg:
                result = OptimizeResult(
                    {
                        'success': True,
                        'x': x,
                        'jac': residual,
                        'nit': n_iterations,
                        'message': 'CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL',
                        })
                if residual_plot:
                    import matplotlib.pyplot as plt
                    plt.plot(iterations, np.log10(gaps))
                    plt.xlabel('iterations')
                    plt.ylabel('residuals')
                    plt.show()

                return result

            elif (n_iterations >= maxiter - 1):
                '''If no convergence'''
                result = OptimizeResult({
                    'success': False,
                    'x': x,
                    'jac': residual,
                    'nit': n_iterations,
                    'message': 'NO-CONVERGENCE: MAXITERATIONS REACHED',
                    })

                if residual_plot:
                    import matplotlib.pyplot as plt
                    plt.plot(iterations, np.log10(gaps))
                    plt.xlabel('iterations')
                    plt.ylabel('residuals')
                    plt.show()

                return result
