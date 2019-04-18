#!/usr/bin/env bash

sudo apt-get install python3-numpy python3-scipy
if [ "$WITH_MPI" == "yes" ]; then
    sudo apt-get install openmpi-bin libopenmpi-dev
    sudo python -m pip install mpi4py
fi