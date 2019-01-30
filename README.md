MPITools
========

Testing
-------

From main source directory

```bash
source env.sh
python setup.py test
mpirun -np <nbofprocessors> python MPI_testrunner.py

```
each processor should tell 
`The following 0 tests Failed: []` near the end.