NuMPI
=====

NuMPI is a collection of numerical tools for MPI-parallelized Python codes. NuMPI presently contains:

- An (incomplete) stub implementation of the [mpi4py](https://github.com/mpi4py/mpi4py) interface to the MPI libraries. This allows running serial versions of MPI parallel code  without having `mpi4py` (and hence a full MPI stack) installed.
- Parallel file IO in numpy's [.npy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html) format using MPI I/O.
- An MPI-parallel implementation of the L-BFGS optimizer.
- An MPI-parallel bound constrained conjugate gradients algorithm.

Build status
------------

[![Tests](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/tests.yml) [![Flake8](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/flake8.yml/badge.svg?branch=master)](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/flake8.yml)

Installation
------------

```
python3 -m pip install NuMPI
```

Development Installation
------------------------

Clone the repository.

To use the code, use the env.sh script to set the environment:

```
source /path/to/NuMPI/env.sh
```

Testing
-------

You have to do a development installation to be able to run the tests.

We use [runtests](https://github.com/bccp/runtests). 

From the main installation directory:
```bash
python run-tests.py
```

If you want to use NuMPI without mpi4py, you can simply run the tests with pytest. 

```bash
pytest tests/
```

Testing on the cluster
----------------------
On NEMO for example

```bash
msub -q express -l walltime=15:00,nodes=1:ppn=20 NEMO_test_job.sh -m bea
```

Development & Funding
---------------------

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within project [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).
