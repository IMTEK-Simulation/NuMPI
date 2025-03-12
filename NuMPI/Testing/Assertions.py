import numpy as np

from NuMPI import MPI


def _assert_one(fun):
    def assert_func(comm, rank, *args, **kwargs):
        exception = None
        if comm.rank == rank:
            try:
                fun(*args, **kwargs)
            except AssertionError as exc:
                exception = exc

        failed_recv = np.ones(1, dtype=int)
        comm.Allreduce(
            np.array([exception is not None], dtype=int), failed_recv, op=MPI.SUM
        )
        if failed_recv[0] > 0:
            if exception is None:
                raise AssertionError()
            else:
                raise exception

    return assert_func


def _assert_all(fun):
    def assert_func(comm, *args, **kwargs):
        exception = None
        try:
            fun(*args, **kwargs)
        except AssertionError as exc:
            exception = exc

        failed_recv = np.ones(1, dtype=int)
        comm.Allreduce(
            np.array([exception is not None], dtype=int), failed_recv, op=MPI.SUM
        )
        if failed_recv[0] > 0:
            if exception is None:
                raise AssertionError()
            else:
                raise exception

    return assert_func


assert_one_array_equal = _assert_one(np.testing.assert_array_equal)
assert_all_array_equal = _assert_all(np.testing.assert_array_equal)

assert_one_allclose = _assert_one(np.testing.assert_allclose)
assert_all_allclose = _assert_all(np.testing.assert_allclose)
