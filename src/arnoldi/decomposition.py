import dataclasses

import numpy as np
import numpy.linalg as nlin


class ArnoldiDecomposition:
    """ Create an arnoldi decomposition for operators of dimension n, with a Krylov
    basis of up to max_dim dimensions.

    Parameters
    ==========
    n: int
        The dimension of each 
    max_dim: int
        Max dimension of the underlying Krylov space.
    """
    def __init__(self, n, max_dim, dtype=np.complex128, atol=None):
        self.n = n
        self.max_dim = max_dim

        self.v = np.zeros((n, max_dim+1), dtype=dtype)
        self.h = np.zeros((max_dim+1, max_dim), dtype=dtype)

        # Logic of sqrt copied from Julia's ArnoldiMethod.jl package
        self.atol = atol or np.sqrt(np.finfo(self._dtype).eps)

    @property
    def _dtype(self):
        return self.h.dtype

    def initialize(self, init_vector=None):
        if init_vector is None:
            init_vector = np.random.randn(self.n).astype(self._dtype)
            init_vector /= np.linalg.norm(init_vector)

        self.v[:, 0] = init_vector

    def iterate(self, a, start_dim=0, atol=None):
        atol = atol or self.atol
        for j in range(start_dim, self.max_dim):
            v = self.v[:, j+1]
            v[:] = a @ self.v[:, j]

            # Modified Gram-Schmidt (orthonormalization)
            for i in range(j + 1):
                self.h[i, j] = np.vdot(self.v[:, i], v)
                v -= self.h[i, j] * self.v[:, i]

            self.h[j + 1, j] = np.linalg.norm(v)

            if self.h[j + 1, j] < self.atol:
                return j
            v /= self.h[j + 1, j]

        return self.max_dim

    def _extract_arnold_decomp(self, k=None):
        """ Return V_k/H_k such as V_k^H A V_k = V_k.
        """
        v, h = self.v, self.h
        k = k or self.max_dim

        v_k = v[:, :k]
        h_k = h[:k, :k]

        return v_k, h_k


@dataclasses.dataclass
class RitzDecomposition:
    values: np.ndarray
    vectors: np.ndarray

    # The approximate residuals
    approximate_residuals: np.ndarray

    @classmethod
    def from_arnoldi(cls, arnoldi, n_ritz, start_dim=0, max_dim=None):
        # n_ritz is the number of ritz values to extract, k is the number of
        # active dimension of the arnoldi decomposition.
        max_dim = max_dim or arnoldi.max_dim
        v_k, h_k = arnoldi._extract_arnold_decomp(max_dim)
        v_active = v_k[:, start_dim:]
        h_active = h_k[start_dim:, start_dim:]

        ritz_values, vectors = _largest_eigvals(h_active, n_ritz)
        ritz_vectors = v_active @ vectors

        return cls(
            ritz_values, ritz_vectors,
            _approximate_residuals(arnoldi.h, vectors, max_dim)
        )

    def compute_residuals(self, a):
        """ The "true" residuals of this ritz decomposition, i.e.

            ||a @ v - lambda * v||

        for the ritz vectors v and ritz values lambda
        """
        return nlin.norm(
            a @ self.vectors - self.values * self.vectors, axis=0
        )


def _approximate_residuals(h, v_k, k):
    # Given a matrix A with arnoldi decomposition A * v = v * h_k + ...,
    # for a given eigen value / vector pair lambda_i / u_i of h_k,
    #
    # ||A u_i - lambda_i u_i|| = h[k, k-1] * |<e_k, v_k>| where u_k = q * v_k
    #
    # See e.g. proposition 6.8 of
    # https://www-users.cse.umn.edu/~saad/eig_book_2ndEd.pdf
    return np.abs(h[k, k-1] * v_k[-1])


def _largest_eigvals(h, n_ev):
    eigvals, eigvecs = nlin.eig(h)
    ind = np.argsort(np.abs(eigvals))[:-n_ev-1:-1]
    return eigvals[ind], eigvecs[:, ind]
