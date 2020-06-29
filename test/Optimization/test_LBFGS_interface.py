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

import numpy as np
import scipy.optimize
import pytest
from NuMPI.Optimization import LBFGS
from NuMPI.Tools import Reduction

import test.Optimization.minimization_problems as mp
import test.Optimization.MPI_minimization_problems as mpi_mp


@pytest.fixture
def pnp(comm):
    return Reduction(comm)


@pytest.mark.parametrize("Objectiveclass", [mpi_mp.MPI_Quadratic])
def test_minimize_call_column(pnp, Objectiveclass):
    """
    Checks the compatibility with a scipy.optimize.minimze call. Note that one
    should not use this type of call when running in parallel

    Parameters
    ----------
    pnp
    Objectiveclass

    Returns
    -------

    """

    comm = pnp.comm
    n = 10 + 2 * comm.Get_size()
    Objective = Objectiveclass(n, pnp=pnp)
    result = scipy.optimize.minimize(Objective.f_grad, Objective.startpoint(),
                                     jac=True, method=LBFGS, options={
            "gtol": 1e-8, "ftol": 1e-20, "pnp": pnp
        })
    assert result.success
    assert pnp.max(abs(
        np.reshape(result.x, (-1,)) - np.reshape(Objective.xmin(), (-1,))
    )) / pnp.max(abs(Objective.startpoint() - Objective.xmin())) < 1e-5


@pytest.mark.parametrize("Objectiveclass", [mpi_mp.MPI_Quadratic])
def test_minimize_call_row(pnp, Objectiveclass):
    comm = pnp.comm
    n = 10 + 2 * comm.Get_size()
    Objective = Objectiveclass(n, pnp=pnp)
    result = scipy.optimize.minimize(
        Objective.f_grad,
        Objective.startpoint().reshape(-1),
        method=LBFGS, jac=True,
        options={"gtol": 1e-8, "ftol": 1e-20, "pnp": pnp})
    assert result.success, ""
    assert pnp.max(abs(
        np.reshape(result.x, (-1,)) - np.reshape(Objective.xmin(), (-1,))
    )) / pnp.max(abs(Objective.startpoint() - Objective.xmin())) < 1e-5


# TODO: test when jac is Bool

@pytest.mark.parametrize("shape", [(10,), (10, 1), (1, 10), (2, 4)])
def test_shape_unchanged(shape):
    """
    This test is only serial
    """
    size = np.prod(shape)
    Objective = mp.Extended_Rosenbrock

    x0 = Objective.startpoint(size).reshape(shape)

    result = LBFGS(Objective.f_grad, x0, jac=True, gtol=1e-10, ftol=1e-40,
                   pnp=np)

    assert result.success, ""
    np.testing.assert_allclose(np.reshape(result.x, (-1,)),
                               np.reshape(Objective.xmin(size), (-1,)),
                               rtol=1e-5)

    assert result.x.shape == shape, \
        "shape of result mismatch shape of startpoint"
    assert result.jac.shape == shape, \
        "shape of result jac mismatch shape of startpoint"


# FIXME: Implement
def test_multiple_tol():
    pass


def test_gtol():
    pass


def test_g2tol():
    pass


def test_ftol():
    pass


def test():
    pass
