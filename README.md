NuMPI
=====

NuMPI is a collection of numerical tools for MPI-parallelized Python codes. NuMPI presently contains:

- An (incomplete) stub implementation of the [mpi4py](https://bitbucket.org/mpi4py/mpi4py) interface to the MPI libraries.
- Parallel file IO in numpy's [.npy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html) format.
- An MPI-parallel implementation of the L-BFGS optimizer.

Testing
-------

We use [runtests](https://github.com/bccp/runtests). 

We had to add some modifications so presently you should install it directly from my fork: 

```
pip install -e  git+git@github.com:AntoineSIMTEK/runtests.git#egg=runtests
```

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
msub -q expresss -l walltime=15:00,nodes=1:ppn=20 NEMO_test_job.sh -m bea
```
