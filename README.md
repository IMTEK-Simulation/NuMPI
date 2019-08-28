NuMPI
=====

NuMPI is a collection of numerical tools for MPI-parallelized Python codes. NuMPI presently contains:

- An (incomplete) stub implementation of the [mpi4py](https://bitbucket.org/mpi4py/mpi4py) interface to the MPI libraries.
- Parallel file IO in numpy's [.npy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html) format.
- An MPI-parallel implementation of the L-BFGS optimizer.

Build status
------------

[![Build Status](https://www.travis-ci.org/IMTEK-Simulation/NuMPI.svg?branch=master)](https://www.travis-ci.org/IMTEK-Simulation/NuMPI)

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
source /path/to/PyCo/env.sh [python3]
```

Testing
-------

You have to do a development installation to be able to run the tests.

We use [runtests](https://github.com/bccp/runtests). At the moment a slight modification in my [fork](https://github.com/AntoineSIMTEK/runtests). 

(To install with pip: `python3 -m pip install https://github.com/AntoineSIMTEK/runtests.git`)

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
