import numpy as np
import numpy.linalg as nlin

from .utils import rand_normalized_vector


norm = nlin.norm


class Arnoldi:
    """ Create an arnoldi solver for operators of dimension n, with a Krylov
    basis of up to m dimensions.
    """
    def __init__(self, n, m, dtype=np.complex128):
        self.n = n
        self.m = m

        self.V = np.zeros((n, m+1), dtype=dtype)
        self.H = np.zeros((m+1, m), dtype=dtype)

    @property
    def _dtype(self):
        return self.H.dtype

    @property
    def _atol(self):
        # Logic of sqrt copied from Julia's ArnoldiMethod.jl package
        return np.sqrt(np.finfo(self._dtype).eps)

    def initialize(self, init_vector=None):
        init_vector = init_vector or rand_normalized_vector(self.n, self._dtype)
        self.V[:, 0] = init_vector

    def iterate(self, A):
        _, _, m = arnoldi_decomp(A, self.V, self.H, self._atol, self.m)
        return m

    def _extract_arnold_decomp(self, m=None):
        """ Return V_m/H_m such as V_m^H A V_m = H_m.
        """
        m = m or self.m
        return self.V[:, :m], self.H[:m, :m]

    def _extract_ritz_decomp_and_raw_base(self, n_ritz, m=None):
        m = m or self.m
        V_m, H_m = self._extract_arnold_decomp(m)

        ritz_values, vectors = _largest_eigvals(H_m, n_ritz)
        ritz_vectors = V_m @ vectors
        return ritz_values, ritz_vectors, vectors

    def _extract_ritz_decomp(self, n_ritz, m=None):
        """ Extract n_ritz ritz values / vectors."""
        ritz_values, ritz_vectors, vectors = self._extract_ritz_decomp_and_raw_base(n_ritz, m)
        return ritz_values, ritz_vectors

    def _approximate_residuals(self, n_ritz, m):
        # For a given eigen value/vector pair lambda_i / u_i, we have:
        #
        # ||A u_i - lambda_i u_i|| = h[k, k-1] * |<e_k, v_k>| where u_k = q * v_k
        #
        # See e.g. proposition 6.8 of
        # https://www-users.cse.umn.edu/~saad/eig_book_2ndEd.pdf
        _, _, V_m = self._extract_ritz_decomp_and_raw_base(n_ritz, m)
        return np.abs(self.H[m, m-1] * V_m[-1])

    def _residuals(self, A, n_ritz, m=None):
        ritz_values, ritz_vectors = self._extract_ritz_decomp(n_ritz, m)
        return nlin.norm(A @ ritz_vectors - ritz_values * ritz_vectors, axis=0)


def _largest_eigvals(H, n_ev):
    eigvals, eigvecs = nlin.eig(H)
    ind = np.argsort(np.abs(eigvals))[:-n_ev-1:-1]
    return eigvals[ind], eigvecs[:, ind]


def arnoldi_decomp(A, V, H, invariant_tol, max_dim=None):
    """Run the arnoldi decomposition for square matrix a of dimension n.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        square matrix to be decomposed
    V : ndarray of shape (n, m+1)
        the krylov basisexpected built by the Arnoldi decomposition
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
    n = A.shape[0]
    m = V.shape[1] - 1

    assert A.shape[1] == n, "A is expected to be square matrix"
    assert V.shape == (n, m+1), "V must has same number of rows as V"
    assert H.shape == (m+1, m), f"H must be {m+1, m}, is {H.shape}"

    max_dim = max_dim or m
    assert max_dim <= m, "max_dim > m violated"

    for j in range(max_dim):
        v = V[:, j+1]
        v[:] = A @ V[:, j]

        # Modified Gram-Schmidt (orthonormalization)
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], v)
            v -= H[i, j] * V[:, i]

        H[j + 1, j] = norm(v)

        if H[j + 1, j] < invariant_tol:
            raise ValueError("Lucky break not supported yet")
        v /= H[j + 1, j]

    return V[:, :max_dim+1], H[:max_dim+1, :max_dim], max_dim
