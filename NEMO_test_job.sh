
#MSUB -l nodes=1:ppn=2
#MSUB -l walltime=15:00
#MSUB -l pmem=2000mb
cd $MOAB_SUBMITDIR

echo $PWD 
module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles/

module load devel/python/3.6.5.20180618

## annulate the module load mpi/openmpi/2.1-gnu-4.8 that was in the above command 
module unload mpi/openmpi/2.1-gnu-4.8

VENVPATH=../pythonvenv

source $VENVPATH/bin/activate

module load mpi4py/3.0.0
source env.sh python3

export LD_PRELOAD=/opt/bwhpc/common/compiler/gnu/5.2.0/lib64/libstdc++.so.6

rm testjob.log
python3 run-tests.py --mpirun="mpirun --bind-to core --map-by core -report-bindings"  >> testjob.log 2>&1
 
