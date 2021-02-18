Change log for NuMPI
===================

v0.1.5. (>210122)
-----------------

- ENH: NetCDF IO : NCStructuredGrid migrated from muspectre to here
- Drop support for Python3.5

v0.1.4 (16Oct20)
----------------
- BUG: don't close filestream on failure

v0.1.3 (14Oct20)
----------------
- BUG: make_mpi_file_view was not compatible with filestreams

v0.1.2 (29Jun20)
----------------
- Remove usage of xrange (#29)

v0.1.1 (02Dec19)
----------------

- Reader from Stub MPI communicator compatible with filestrem input 

v0.1.0 (25Aug19)
----------------

- Initial release
- Stub MPI communicator
- Helper functions for make reduction operations more numpy-like
- MPI-parallel IO of `npy` files
- MPI-parallel L-BFGS
