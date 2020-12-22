import numpy as np
import scipy.optimize as optim
from inspect import signature


def min_cg(objective, hessp, x0=None, gtol=1e-8, mean_value=None,
           bugnicourt=False, polonskykeer=False,
           residual_plot=False, maxiter=5000, logger=None, callback=None):
    """
    A generic Constrained Conjugate Gradient Algorithm implementation with
    multiple options
    to switch between some standard implementations of Conjugate gradient
    algortihm developed overtime
    to solve the "problem with rough contact surfaces with/without adhesion."

    Parameters
    ----------
    objective : callable.
                The objective function to be minimized.
                            fun(x) -> float(energy),ndarray(gradient)
                where x is the input ndarray.
    hessp : callable
            Function to evaluate the hessian product of the objective.
            Hessp should accept either 1 argument (desscent direction) or
            2 arguments (x,descent direction).
                            hessp(des_dir)->ndarray
                                    or
                            hessp(x,des_dir)->ndarray
            where x is the input ndarray and des_dir is the descent direction.

    x0 : ndarray
         Initial guess. Default value->None.
         ValueError is raised if "None" is provided.

    gtol : float, optional
           Default value : 1e-8

    mean_value : int/float, optional
               If you want to apply the mean_value constraint then provide an
               int/float value to the mean_value.

    bugnicourt : bool, optional
                 If you want to use the algorithm developed by [
                 1]-Bugnicourt et.al-2018

    polonskykeer : bool, optional
                   If you want to use the algorithm developed by [
                   2]-Polonsky & Keer et.al-1999
                   or the version for primal problem by [3]-Molinari & Rey
                   et.al-2017

    residual_plot : bool, optional
                    Generates a plot between the residual and iterations.

    maxiter : int,optional
              Default, maxiter=5000
              Maximum number of iterations after which the program will exit.

    Returns
    -------
    success : bool
              True if convergence else False.
    x : array
        array of minimized x.
    jac : array
          value of residual at convergence/non-convergence.
    nit : int
          Number of iterations
    message : string
              Convergence or Nodes_dir-Convergence

    References
    ----------
    ..[1] Bugnicourt et al.2018. FFT-Based Methods for Solving a Rough Adhesive
          Contact: Description and Convergence Study
    ..[2] Polonsky and Keer 1999 - A numerical method for solving rough
    contact problems based on the
          multi-level multi-summation and conjugate gradient techniques
    ..[3] Rey et al. - 2017 - Normal adhesive contact on rough surfaces :
    efficient algorithm
          for FFT-based BEM resolution
    """

    if not isinstance(mean_value, (type(None), int, float)):
        raise ValueError('Inappropriate type: {} for mean_value whereas a '
                         'float \
            or int is expected'.format(type(mean_value)))

    if not isinstance(residual_plot, bool):
        raise ValueError('Inappropriate type: {} for "residual_plot" whereas '
                         'a bool \
                         is expected'.format(type(residual_plot)))

    if np.max(abs(x0)) == 0:
        raise ValueError('This is not a reasonable initial guess. x0>0 !!')

    gtol = gtol
    fun = objective
    hessp = hessp

    x0 = x0.flatten()

    if x0 is not None:
        x = x0.copy()
        if polonskykeer:
            '''Take care of constraint a_x*a_y*sum(p_ij)=P0'''
            # print("Total force is  :: {}".format(np.sum(x)))
            delta = 0
            G_old = 1
            delta_str = 'sd'
    else:
        raise ValueError('Input required for x0/initial value !!')

    mask_c = x > 0

    'Mask to truncate the negative values'
    # mask_not_c = x <= 0  # TODO: find a more explicit name, like active set:
    # where the constraint is actuve. Not sure what is the appropriate name
    x[np.logical_not(mask_c)] = 0.0

    'Initial residual or GAP for polonsky-keer'
    residual = fun(x)[1]
    residual = residual

    if mean_value is not None:
        residual = residual - np.mean(residual[mask_c])

    if polonskykeer:
        mask_c = x > 0
        # if mean_value is not None:
        #     residual = residual - np.mean(residual[mask_c])
        G = np.sum(residual[mask_c] ** 2)
    else:
        mask_res = residual > 0
        mask_bounded = np.logical_and(np.logical_not(mask_c), mask_res)
        residual[mask_bounded] = 0.0

    residual_mag = np.sum(residual ** 2) ** (1.0 / 2.0)
    des_dir = np.zeros(np.shape(x))

    if polonskykeer:

        des_dir[mask_c] = -residual[mask_c]
        des_dir[np.logical_not(mask_c)] = 0

        G_old = G
        if np.logical_not(mask_c).sum() > 0:
            rms_pen = np.sqrt(G / np.logical_not(mask_c).sum())
        else:
            rms_pen = np.sqrt(G)  # TODO: I guess in this case G is one anyway.
            #       Is this line actually ever executed ?
    else:
        '''Set descent direction initially to steepest descent'''
        des_dir[np.logical_not(mask_bounded)] = -residual[
            np.logical_not(mask_bounded)]

    residual_old = residual
    des_dir_old = des_dir
    res_mag_old = residual_mag

    grads = np.array([])
    converged = False
    res_convg = False

    if residual_plot:
        if polonskykeer:
            if mask_c.sum() != 0:
                grads = np.append(grads, np.max(abs(residual[mask_c])))
            else:
                grads = np.append(grads, np.max(abs(residual)))
            rms_pen_ = np.array([])
            rms_pen_ = np.append(rms_pen_, rms_pen)

        else:
            grads = np.append(grads, np.max(abs(residual)))

    n_iterations = 1

    if callback is not None:
        mask_c = x > 0
        d = dict(gradient=residual,
                 frac_ar=mask_c.sum() / len(x))
        callback(n_iterations, x, d)

    while (n_iterations <= maxiter + 1):

        '''Calculating alpha'''
        sig = signature(hessp)
        if len(sig.parameters) == 2:
            hessp_val = hessp(x, des_dir)
        elif len(sig.parameters) == 1:
            hessp_val = hessp(des_dir)
        else:
            raise ValueError('hessp function has to take max 1 arg (descent '
                             'dir)\
                            or 2 args (x, descent direction)')

        denom = np.sum(des_dir.T * hessp_val)

        if bugnicourt and not polonskykeer:
            alpha = -np.sum(residual.T * des_dir) / denom
        elif not bugnicourt and polonskykeer:
            if mean_value is not None:
                '''Here hessp_val is used as r_ij in original algorithm'''
                hessp_val = hessp_val - np.mean(hessp_val[mask_c])
            if mask_c.sum() != 0:
                '''alpha is TAU from algorithm'''
                alpha = -np.sum(residual[mask_c] * des_dir[mask_c]) / np.sum(
                    hessp_val[mask_c] * des_dir[mask_c])
            else:
                alpha = 0.0
            assert alpha >= 0
        else:
            alpha = np.sum(residual.T * residual) / denom

        if polonskykeer and not bugnicourt:
            x[mask_c] += alpha * des_dir[mask_c]
        else:
            '''Updating the objective variable'''
            x += alpha * des_dir

        '''mask for contact'''
        mask_c = x > 0
        '''truncating negative values'''
        x[np.logical_not(mask_c)] = 0.0

        '''If mean gap is constant then enforce an extra condition on gap "x".
         '''
        if mean_value is not None and not polonskykeer:
            x = (mean_value / np.mean(x)) * x

        if not polonskykeer:
            '''Updating old residual'''
            residual_old = residual

            '''Updating the new residual as gradient of objective'''
            residual = fun(x)[1]

        '''If mean gap is constant then enforce an extra condition on residual
        '''
        if mean_value is not None and not polonskykeer:
            mask_c = x > 0
            residual = residual - np.mean(residual[mask_c])

        if polonskykeer:
            mask_res = residual < 0
            mask_overlap = np.logical_and(np.logical_not(mask_c), mask_res)
            if mask_overlap.sum() == 0:
                delta = 1
                delta_str = 'cg'
            else:
                delta = 0
                x[mask_overlap] = x[mask_overlap] - alpha * residual[
                    mask_overlap]
                delta_str = 'sd'
        else:
            mask_res = residual > 0
            '''Updating residual as per NA region'''
            mask_bounded = np.logical_and(np.logical_not(mask_c), mask_res)
            residual[mask_bounded] = 0.0

        '''If mean gap is constant then enforce an extra condition on residual
        '''
        if mean_value is not None and not polonskykeer:
            mask_nonzero = residual != 0
            residual[mask_nonzero] = residual[mask_nonzero] - np.mean(
                residual[mask_nonzero])

        # '''updating old residual magnitude value and new one'''
        res_mag_old = residual_mag
        residual_mag = np.sum(residual ** 2) ** (1.0 / 2.0)

        if polonskykeer:
            if mean_value is not None:
                x *= (mean_value / np.mean(x))
            mask_neg = x <= 0
            x[mask_neg] = 0.0
            residual = fun(x)[1]
            # residual = residual.reshape(np.shape(x))
            mask_c = x > 0
            if mean_value is not None:
                residual = residual - np.mean(residual[mask_c])
            G = np.sum(residual[mask_c] ** 2)

        if mean_value is not None and not bugnicourt and not polonskykeer:
            beta = np.sum(residual.T * residual) / np.sum(
                residual_old.T * residual_old)
        elif bugnicourt and not polonskykeer:
            beta = np.sum(residual.T * (residual - residual_old)) / np.sum(
                alpha * denom)
            # beta = np.sum(residual.T * hessp_val) / denom
        elif polonskykeer and not bugnicourt:
            '''Take care of constraint a_x*a_y*sum(p_ij)=P0'''
            beta = delta * (G / G_old)
        else:
            # beta = np.sum(residual.T * (residual - residual_old)) / denom
            '''Use this beta whe solving primal!!!'''
            beta = -np.sum(
                residual.T * (residual - residual_old)) / res_mag_old

        des_dir_old = des_dir

        if not bugnicourt and polonskykeer:
            des_dir[mask_c] = -residual[mask_c] + beta * des_dir[mask_c]
            des_dir[np.logical_not(mask_c)] = 0
            G_old = G
            if np.logical_not(mask_c).sum() > 0:
                rms_pen = np.sqrt(G / np.logical_not(mask_c).sum())
            else:
                rms_pen = np.sqrt(G)
        else:
            des_dir = -residual + beta * des_dir_old
            des_dir[np.logical_not(mask_bounded)] = -residual[
                np.logical_not(mask_bounded)] + beta * des_dir_old[
                                                        np.logical_not(
                                                            mask_bounded)]
            des_dir[mask_bounded] = 0

        if residual_plot:
            if polonskykeer:
                if mask_c.sum() != 0:
                    grads = np.append(grads, np.max(abs(residual[mask_c])))
                else:
                    grads = np.append(grads, np.max(abs(residual)))
                rms_pen_ = np.append(rms_pen_, rms_pen)
            else:
                grads = np.append(grads, np.max(abs(residual)))

        n_iterations += 1
        res = np.max(abs(residual))

        assert np.logical_not(np.isnan(x).any())

        if polonskykeer:
            frac_ar = mask_c.sum() / len(x)

            log_headers = ['status', 'it', 'frac_area', 'gradient']
            log_values = [delta_str, n_iterations, frac_ar, res]
        else:
            log_headers = ['it', 'gradient']
            log_values = [n_iterations, res]

        if callback is not None:
            mask_c = x > 0
            frac_ar = mask_c.sum() / len(x)
            d = dict(gradient=residual,
                     frac_ar=frac_ar)
            callback(n_iterations, x, d)

        if logger is not None:
            logger.st(log_headers, log_values, force_print=True)

        if polonskykeer:
            if mask_c.sum() != 0:
                if np.max(abs(residual[mask_c])) <= gtol:
                    res_convg = True
                else:
                    res_convg = False
            if (res_convg or (rms_pen <= gtol)) and n_iterations >= 10:
                converged = True

                if residual_plot:
                    import matplotlib.pyplot as plt
                    plt.plot(range(n_iterations), np.log10(grads),
                             label='residuals')
                    plt.plot(range(n_iterations), np.log10(rms_pen_),
                             label='rms_pen')
                    plt.xlabel('iterations')
                    plt.ylabel('residuals')
                    plt.legend()
                    plt.show()

            else:
                converged = False

        elif (np.max(abs(residual)) <= gtol) and n_iterations >= 3:
            converged = True
            rms_pen_ = 0.0
            if residual_plot:
                import matplotlib.pyplot as plt
                plt.plot(range(n_iterations), np.log10(grads),
                         label='residuals')
                plt.xlabel('iterations')
                plt.ylabel('residuals')
                plt.show(block=True)

        if converged:

            if logger is not None:
                log_values[0] = 'CONVERGED'
                logger.st(log_headers, log_values, force_print=True)

            result = optim.OptimizeResult({'success': True,
                                           'x': x,
                                           'fun': fun(x)[0],
                                           'jac': residual,
                                           'nit': n_iterations,
                                           'message':
                                               'CONVERGENCE: '
                                               'NORM_OF_GRADIENT_<=_GTOL',
                                           })
            return result

        elif n_iterations >= maxiter - 1:
            '''If no convergence'''
            print(
                '####MAXITERATIONS######MAXITERATIONS####MAXITERATIONS'
                '#####MAXITERATIONS#####')
            result = optim.OptimizeResult({'success': False,
                                           'x': x,
                                           'fun': fun,
                                           'jac': residual,
                                           'nit': n_iterations,
                                           'message': 'NO-CONVERGENCE:',
                                           })

            if logger is not None:
                log_values[0] = 'NOT CONVERGED'
                logger.st(log_headers, log_values, force_print=True)

            if residual_plot:
                import matplotlib.pyplot as plt
                plt.plot(range(n_iterations), np.log10(grads),
                         label='residuals')
                plt.xlabel('iterations')
                plt.ylabel('residuals')
                plt.show(block=True)

            return result
