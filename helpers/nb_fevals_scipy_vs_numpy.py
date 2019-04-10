"""
This runs LBFGS and scipy L-BFGS-B in serial and compare the number of iterations needed
"""

import os
from NuMPI.Optimization import LBFGS
from tests.minimization_problems import Trigonometric, Extended_Rosenbrock
import numpy as np
import scipy

import matplotlib.pyplot as plt

#import sys
n = 10**3

class decorated_objective:
    def __init__(self,  objective):
        self.objective = objective
        self.neval = 0
        self.energies = []
        self.maxgradients = []

    def __call__(self, *args, **kwargs):
        f, grad = self.objective(*args, **kwargs)
        self.neval += 1
        self.energies.append(f)
        self.maxgradients.append(np.max(abs(grad)))
        return f, grad



for Objective in [Trigonometric, Extended_Rosenbrock]:
    fig, (axEn, axgrad) = plt.subplots(2, 1, sharex=True)
    for method, name, c  in zip([LBFGS, "L-BFGS-B"], ["NuMPI", "scipy"], ["C1", "C0"]):
        objective_monitor = decorated_objective(Objective.f_grad)
        result = scipy.optimize.minimize(objective_monitor, Objective.startpoint(n),
                                         jac=True,method=method,
                                         options ={"gtol":1e-6,"ftol":1e-20, "maxcor":20})
        assert result.success



        axgrad.plot(range(objective_monitor.neval),
                    objective_monitor.maxgradients, label="{}".format(name))
        axEn.plot(range(objective_monitor.neval), (
                    np.array(objective_monitor.energies) - objective_monitor.energies[
                -1]) / (objective_monitor.energies[0] -
                        objective_monitor.energies[-1]),
                  label="{}".format(name), c=c)

        print("{}, {}: nevals {}, nit {}, evals/it {}".format(name,
                                                              Objective.__name__,
                                                              objective_monitor.neval,
                                                              result.nit, float(
                objective_monitor.neval) / result.nit))


    axEn.set_xlabel("# of objective evaluations")
    axEn.set_ylabel("E(i)-E(last) / (E(0)-E(last))")
    axEn.set_yscale("log")

    axgrad.set_yscale("log")
    axgrad.set_ylabel(r"$|grad|_{\infty}$")
    axgrad.legend()

    for a in (axEn, axgrad):
        a.set_xlabel("# of objective evaluations")
        a.label_outer()

    fig.suptitle("{}, n={}".format(Objective.__name__, n))
    fig.savefig("{}_{}.png".format(os.path.basename(__file__), Objective.__name__))