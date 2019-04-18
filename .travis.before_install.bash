#!/usr/bin/env bash

if [ "$WITH_MPI" == "yes" ]; then
    sudo apt-get install openmpi-bin libopenmpi-dev
    sudo python -m pip install mpi4py
fi