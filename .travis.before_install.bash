#!/usr/bin/env bash

sudo apt-get update
if [ "$WITH_MPI" == "yes" ]; then
  sudo apt-get install openmpi-bin libopenmpi-dev libfftw3-mpi-dev
fi
python -m pip install $(grep numpy requirements.txt)
if [ "$WITH_MPI" == "yes" ]; then
  python -m pip install --no-binary mpi4py mpi4py==${MPI4PY_VERSION}
  BUILDDIR=/tmp PREFIX=$HOME/.local source .install_parallel_netcdf.sh
fi
