MPITools
========

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

If you want to use MPITools without mpi4py, you can simply run the tests with pytest. 

```bash
pytest tests/
```