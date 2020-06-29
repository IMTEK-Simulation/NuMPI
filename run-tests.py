import sys
import os.path

from runtests.mpi import Tester

tester = Tester(os.path.join(os.path.abspath(__file__)), "NuMPI")

print(Tester.__doc__)

tester.main(sys.argv[1:])
