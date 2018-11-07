import numpy as np

import scipy.optimize


def fun(x):
    return np.asscalar(x[0]**2+x[1]**2)
def der(x):
    return 2 * x


x0 = np.array([2,1])
#x0.shape = (-1,1)  # the linesearch only works with 1D arrays


print(x0)
print(x0.shape)

pk = -der(x0)
#scipy.optimize.line_search()
result = scipy.optimize.line_search(fun,der,x0,pk)
print(result)
#raise NotImplementedError

