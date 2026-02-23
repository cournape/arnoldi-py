import petsc4py, slepc4py
petsc4py.init()
slepc4py.init()
from slepc4py import SLEPc
eps = SLEPc.EPS().create()
print("petsc4py:", petsc4py.__version__)
print("slepc4py:", slepc4py.__version__)
print("SLEPc EPS created OK")
eps.destroy()
