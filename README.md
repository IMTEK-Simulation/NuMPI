MPITools
========

Testing
-------

We use [runtests](https://github.com/bccp/runtests). 


From the main installation directory:
```bash
python run-tests.py
```

If you want to use MPITools without mpi4py, you can simply run the tests with pytest. 

```bash
pytest tests/
```