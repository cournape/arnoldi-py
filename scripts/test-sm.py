import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import splu, LinearOperator

from arnoldi import partial_schur


K = 5
MAX_DIMS = 20
P = 10
TOL = 1e-8

N = 200
A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format='csc').astype(np.complex128)
r_vals = np.sort(2 - 2 * np.cos(np.arange(1, K+1) * np.pi / (N + 1)))

# --- Shift-invert preprocessing ---
sigma = 0.0  # shift of 0 for SM; change to target other regions
N = A.shape[0]

# 1. Form the shifted operator and factor it
A_shifted = A - sigma * sp.eye(N, format='csc')
LU = splu(A_shifted.tocsc())  # sparse LU factorization

# 2. Wrap (A - σI)^{-1} as a LinearOperator
OP_inv = LinearOperator((N, N), matvec=LU.solve, dtype=A.dtype)

# 3. Run partial_schur on the INVERTED operator, asking for LM
Q, T, history = partial_schur(
    OP_inv, 
    nev=K, 
    max_dim=MAX_DIMS,
    p=P,
    stopping_criterion=TOL,
)

# 4. Back-transform eigenvalues:  λ_original = σ + 1/μ
#eigenvalues = sigma + 1.0 / eigenvalues_transformed
vals = sigma + 1.0 / np.diag(T[:K, :K])
print(np.allclose(vals, r_vals, rtol=TOL))
