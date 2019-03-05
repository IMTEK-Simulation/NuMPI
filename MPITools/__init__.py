import warnings

try:
    from mpi4py import MPI
except ImportError:
    warnings.warn('Could not import mpi4py; falling back to MPI stub implementation.', ImportWarning)
    import MPITools.MPIStub as MPI

from . import Tools, Optimization