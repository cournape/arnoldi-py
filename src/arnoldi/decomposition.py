import dataclasses

import numpy as np
import numpy.linalg as nlin


class ArnoldiDecomposition:
    """ Create an arnoldi decomposition for operators of dimension n, with a Krylov
    basis of k dimensions.
    """
    def __init__(self, n, k, dtype=np.complex128, atol=None):
        self.n = n
        self.k = k

        self.q = np.zeros((n, k+1), dtype=dtype)
        self.h = np.zeros((k+1, k), dtype=dtype)

        # Logic of sqrt copied from Julia's ArnoldiMethod.jl package
        self.atol = atol or np.sqrt(np.finfo(self._dtype).eps)

    @property
    def _dtype(self):
        return self.h.dtype

    def initialize(self, init_vector=None):
        if init_vector is None:
            init_vector = np.random.randn(self.n).astype(self._dtype)
            init_vector /= np.linalg.norm(init_vector)

        self.q[:, 0] = init_vector

    def iterate(self, a, atol=None):
        atol = atol or self.atol
        for j in range(self.k):
            v = self.q[:, j+1]
            v[:] = a @ self.q[:, j]

            # Modified Gram-Schmidt (orthonormalization)
            for i in range(j + 1):
                self.h[i, j] = np.vdot(self.q[:, i], v)
                v -= self.h[i, j] * self.q[:, i]

            self.h[j + 1, j] = np.linalg.norm(v)

            if self.h[j + 1, j] < self.atol:
                return j
            v /= self.h[j + 1, j]

        return self.k

    def _extract_arnold_decomp(self, size=None):
        """ Return Q_k/H_k such as Q_k^H A Q_k = H_k.
        """
        q, h = self.q, self.h
        size = size or self.k

        q_k = q[:, :size]
        h_k = h[:size, :size]

        return q_k, h_k


@dataclasses.dataclass
class RitzDecomposition:
    values: np.ndarray
    vectors: np.ndarray

    # The approximate residuals
    approximate_residuals: np.ndarray

    @classmethod
    def from_arnoldi(cls, arnoldi, n_ritz, k=None):
        # n_ritz is the number of ritz values to extract, k is the number of
        # active dimension of the arnoldi decomposition.
        k = k or arnoldi.k
        q_k, h_k = arnoldi._extract_arnold_decomp(k)

        ritz_values, vectors = _largest_eigvals(h_k, n_ritz)
        ritz_vectors = q_k @ vectors

        return cls(
            ritz_values, ritz_vectors,
            _approximate_residuals(arnoldi.h, vectors, k)
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
    # Given a matrix A with arnoldi decomposition A * q = q * h_k + ...,
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
