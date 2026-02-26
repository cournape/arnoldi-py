import dataclasses

import numpy as np
import numpy.linalg as nlin

from .ortho import dgks_gs
from .utils import arg_largest_magnitude


norm = nlin.norm


def arnoldi_decomposition(A, V, H, invariant_tol=None, *, start_dim=0, max_dim=None):
    """Run the arnoldi decomposition for square matrix a of dimension n.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        square matrix to be decomposed
    V : ndarray of shape (n, m+1)
        the krylov basis built by the Arnoldi decomposition
    H : ndarray of shape (m+1, m)
        H[:m, :m] is the Hessenberg matrix built by the Arnoldi decomposition
    invariant_tol: float
        threshold that contols when the decomposition will detect the Arnoldi
        decomposition detected an invariant, i.e. when a new vector in Arnoldi
        decomposition has a norm below that threshold.
    max_dim : int
        Max dimension of the Krylov space. By default, guessed from V.shape

    Returns
    -------
    Va : ndarray of shape (n, max_dim+1)
    Ha : ndarray of shape (max_dim+1, max_dim)
    n_iter : int
        <= max_dim. The number of iterations actually run. Is lower than
        max_dim in case a "lucky break" is found, i.e. the Krylov basis
        invariant is lower dimension than max_dim
    """
    # Logic of sqrt copied from Julia's ArnoldiMethod.jl package
    if invariant_tol is None:
        invariant_tol = np.sqrt(np.finfo(A.dtype).eps)

    n = A.shape[0]
    m = V.shape[1] - 1

    assert A.shape[1] == n, "A is expected to be square matrix"
    assert V.shape == (n, m+1), "V must have the same number of rows as A"
    assert H.shape == (m+1, m), f"H must be {m+1, m}, is {H.shape}"

    if max_dim is None:
        max_dim = m

    assert max_dim <= m, "max_dim > m violated"

    for j in range(start_dim, max_dim):
        w = V[:, j+1]
        w[:] = A @ V[:, j]

        beta, breakdown = dgks_gs(w, V[:, :j+1], H[:j+1, j], invariant_tol)
        if breakdown:
            max_dim = j + 1
            return V[:, :max_dim+1], H[:max_dim+1, :max_dim], max_dim
        else:
            H[j + 1, j] = beta
            w /= beta

    return V[:, :max_dim+1], H[:max_dim+1, :max_dim], max_dim


@dataclasses.dataclass
class RitzDecomposition:
    values: np.ndarray
    vectors: np.ndarray

    # The approximate residuals
    approximate_residuals: np.ndarray

    @classmethod
    def from_v_and_h(cls, V, H, n_ritz, *, max_dim=None, sort_function=None):
        """
        Compute the ritz decomposition for the Arnoldi decomposition V and H. Assumes
        A * V[:, :m] = V[:, :m] * H[:m, :m] + H[m, m-1] * (V[:, m] * e_m^H)

        Parameters
        ----------
        V : ndarray of shape (n, max_dim+1)
            The orthonormal basis of the matrix from Arnoldi
        H : ndarray of shape (max_dim+1, max_dim)
            The upper Hessenberg matrix decomposed from Arnoldi
        n_ritz : int
            The number of ritz values/vectors to extract
        max_dim : int, optional
            If given, the number of vectors to consider in V. If not given,
            assumed to be the number of V's columns
        """
        # For a given ritz value/vector pair lambda_i / u_i, we have
        #
        #   ||A u_i - lambda_i u_i|| = h[m, m-1] * |<e_m, s_i^m>|
        #
        # where u_i = V_m * s_i^m is ritz vector for ritz value lambda_i and
        # e_m the m^th vector basis, in other words <e_m, s_i^m> is the last
        # component of s_i
        #
        # In practice, this is may not hold for complex cases, which is why we
        # keep the right hand side in the attribute approximate_residuals
        max_dim = max_dim or V.shape[1] - 1

        assert H.shape[0] > max_dim
        assert H.shape[1] >= max_dim
        assert V.shape[1] > max_dim
        assert n_ritz <= max_dim

        V_m = V[:, :max_dim]
        H_m = H[:max_dim, :max_dim]

        if sort_function is None:
            sort_function = arg_largest_magnitude

        eigvals, eigvecs = nlin.eig(H_m)
        ind = sort_function(eigvals)[:n_ritz]
        S = eigvecs[:, ind]

        ritz_values = eigvals[ind]
        ritz_vectors = V_m @ S

        approximate_residuals = np.abs(H[max_dim, max_dim-1] * S[-1])
        return cls(
            ritz_values,
            ritz_vectors,
            approximate_residuals,
        )

    def compute_true_residuals(self, A):
        """The "true" residuals of this ritz decomposition, i.e.

            res[i] = ||A @ v_i - lambda_i * v_i||

        for the ritz vector v_i and ritz value lambda_i

        Notes
        -----
        This is expensive as it requires projection back into the original
        matrix A
        """
        return norm(A @ self.vectors - self.values * self.vectors, axis=0)
