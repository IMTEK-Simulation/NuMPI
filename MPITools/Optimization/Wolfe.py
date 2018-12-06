import numpy as np

def first_wolfe_condition(fun, x0, fprime, direction, alpha, beta1):
    """
    p. 268, 11.19

    Keyword Arguments:
    fun         -- objective function to minimize
    x0          -- initial guess for solution
    fprime      -- Jacobian (gradient)
    direction   -- search direction (column vec)
    alpha       -- step size
    beta1       -- lower wolfe bound
    """
    return fun(x0+alpha*direction) <= fun(x0) + \
        alpha * beta1 * float(np.dot(fprime(x0).T ,direction))

def second_wolfe_condition(x0, fprime, direction, alpha, beta2):
    """
    p. 270, 11.21

    Keyword Arguments:
    x0        -- initial guess for solution
    fprime    -- Jacobian (gradient) of objective function
    direction -- search direction
    alpha     -- step size
    beta2     -- upper wolfe bound
    """
    return (float(np.dot(fprime(x0 + alpha*direction).T ,direction)) >=
            beta2*float(np.dot(fprime(x0).T , direction) ))