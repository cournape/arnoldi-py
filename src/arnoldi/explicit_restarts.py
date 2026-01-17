import dataclasses

import numpy as np


from .decomposition import RitzDecomposition, arnoldi_decomposition
from .utils import arg_largest_magnitude, rand_normalized_vector


norm = np.linalg.norm


@dataclasses.dataclass
class History:
    matvecs: np.ndarray
    restarts: np.ndarray

    @classmethod
    def from_k(cls, k):
        return cls(np.zeros(k, np.int32), np.zeros(k, np.int32))

    @property
    def k(self):
        return self.matvecs.shape[0]

    @property
    def total_matvecs(self):
        return self.matvecs.sum()


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
            if residuals[0] / max(np.abs(ritz.values[0]), tol) < tol:
                return ritz, True, i
        # FIXME: should take the ritz vector w/ the lowest residual
        v0 = ritz.vectors[:, 0]

    return ritz, False, max_restarts


def mgs(basis, w, tol):
    """Simple modified Gram Schmidt procedure.

    w is modified in-place.
    """
    m = basis.shape[1]
    for j in range(m):
        corr = np.vdot(basis[:, j], w)
        w -= corr * basis[:, j]

    beta = norm(w)
    assert beta > tol, "MGS: Too small norm when orthornormalizing"

    w /= beta
    return w


def explicit_restarts_with_deflation(
    A, nev, *, max_dim=None, stopping_criterion=None, max_restarts=100,
    sort_function=None,
):
    if stopping_criterion is None:
        tol = np.sqrt(np.finfo(A.dtype).eps)
    else:
        tol = stopping_criterion

    if sort_function is None:
        sort_function = arg_largest_magnitude

    assert max_restarts > 0

    n = A.shape[0]
    assert A.shape[1] == n

    if max_dim is None:
        max_dim = min(max(2 * nev + 1, 20), n)

    dtype = np.complex128

    V = np.zeros((n, max_dim+1), dtype=dtype)
    H = np.zeros((max_dim+1, max_dim), dtype=dtype)

    history = History.from_k(nev)

    eivals = np.zeros(nev, dtype=dtype)
    eivecs = np.zeros((n, nev), dtype=dtype)

    for k in range(nev):
        v0 = rand_normalized_vector(n, dtype)
        mgs(V[:, :k], v0, tol)
        V[:, k] = v0

        for restart in range(max_restarts):
            V_a, H_a, n_iter = arnoldi_decomposition(
                A, V, H, start_dim=k, invariant_tol=tol
            )

            m = n_iter
            assert m > k

            if m != max_dim:
                happy_breakdown = True
            else:
                happy_breakdown = False

            matvecs = restart * (max_dim - k) + (m - k)

            V_k = V_a[:, k:]
            H_k = H_a[k:, k:]

            ritz = RitzDecomposition.from_v_and_h(
                V_k, H_k, m - k, sort_function=sort_function
            )

            v_k = ritz.vectors[:, 0]
            lambda_k = ritz.values[0]

            V[:, k] = v_k
            mgs(V[:, :k], V[:, k], tol)

            approximate_residuals = ritz.approximate_residuals
            approximate_convergence = approximate_residuals / np.abs(ritz.values)

            has_converged = happy_breakdown or (approximate_convergence[0] < tol)

            if has_converged:
                for i in range(k+1):
                    H[i, k] = np.vdot(V[:, i], A @ V[:, k])
                H[k+1:-1, k] = 0

                eivals[k] = lambda_k
                eivecs[:, k] = v_k

                history.matvecs[k] = matvecs
                history.restarts[k] = restart + 1
                break
        else:
            raise ValueError(f"Could not converge for value {k}")

    # FIXME:: re-calculating eigenpairs from the final H is often superfluous,
    # but is critical for some matrices such as markov, even tiny ones like
    # mark(10). Not sure why, this does not seem to be documented in the
    # literature I reviewed.
    eivals, Y = np.linalg.eig(H[:nev, :nev])
    eivecs = V[:, :nev] @ Y
    return eivals, eivecs, history
