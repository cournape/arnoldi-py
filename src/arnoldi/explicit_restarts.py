import numpy as np


from .decomposition import RitzDecomposition, arnoldi_decomposition
from .utils import rand_normalized_vector


def naive_explicit_restarts(A, m=None, *, stopping_criterion=None, max_restarts=10):
    if stopping_criterion is None:
        tol = np.sqrt(np.finfo(A.dtype).eps)
    else:
        tol = stopping_criterion

    dtype = np.promote_types(A.dtype, np.complex64)

    n = A.shape[0]
    k = 1  # Naive arnoldi w/o restart only really works for 1 eigenvalue

    if m is None:
        m = min(max(2 * k + 1, 20), n)

    V = np.zeros((n, m+1), dtype)
    H = np.zeros((m+1, m), dtype)

    v0 = rand_normalized_vector(n).astype(dtype)
    for i in range(max_restarts):
        V[:, 0] = v0
        V, H, n_iter = arnoldi_decomposition(A, V, H)
        ritz = RitzDecomposition.from_v_and_h(V, H, k)
        if ritz.approximate_residuals[0] < tol:
            residuals = ritz.compute_true_residuals(A)
            if residuals[0] / max(ritz.values[0], tol) < tol:
                return ritz, True, i
        # FIXME: should take the ritz vector w/ the lowest residual
        v0 = ritz.vectors[:, 0]

    return ritz, False, max_restarts
