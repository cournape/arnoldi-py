import petsc4py
import slepc4py
petsc4py.init()
slepc4py.init()
from slepc4py import SLEPc
from petsc4py import PETSc
eps = SLEPc.EPS().create()
print("petsc4py:", petsc4py.__version__)
print("slepc4py:", slepc4py.__version__)
print(f"Supported dtype is {PETSc.ScalarType}")
print("SLEPc EPS created OK")
eps.destroy()
