#!/usr/bin/env bash
export PYTHONPATH=$HOME/runtests:$HOME/NuMPI:$PYTHONPATH
python3 run-tests.py --mpirun="mpirun -np 2" -s "$@"