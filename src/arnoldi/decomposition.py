import numpy as np
import numpy.linalg as nlin


class Arnoldi:
    """ Create an arnoldi solver for operators of dimension n, with a Krylov
    basis of k dimensions.
    """
    def __init__(self, n, k, dtype=np.complex128):
        self.n = n
        self.k = k

        self.q = np.zeros((n, k+1), dtype=dtype)
        self.h = np.zeros((k+1, k), dtype=dtype)

    @property
    def _dtype(self):
        return self.h.dtype

    @property
    def _atol(self):
        # Logic of sqrt copied from Julia's ArnoldiMethod.jl package
        return np.sqrt(np.finfo(self._dtype).eps)

    def initialize(self, init_vector=None):
        if init_vector is None:
            init_vector = np.random.randn(self.n).astype(self._dtype)
            init_vector /= np.linalg.norm(init_vector)

        self.q[:, 0] = init_vector

    def iterate(self, a, tol=None):
        tol = tol or self._atol
        for j in range(self.k):
            v = self.q[:, j+1]
            v[:] = a @ self.q[:, j]

            # Modified Gram-Schmidt (orthonormalization)
            for i in range(j + 1):
                self.h[i, j] = np.vdot(self.q[:, i], v)
                v -= self.h[i, j] * self.q[:, i]

            self.h[j + 1, j] = np.linalg.norm(v)

            if self.h[j + 1, j] < self._atol:
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

    def _extract_ritz_decomp_and_raw_base(self, n_ritz, k=None):
        k = k or self.k
        q_k, h_k = self._extract_arnold_decomp(k)

        ritz_values, vectors = _largest_eigvals(h_k, n_ritz)
        ritz_vectors = q_k @ vectors
        return ritz_values, ritz_vectors, vectors

    def _extract_ritz_decomp(self, n_ritz, k=None):
        """ Extract n_ritz ritz values / vectors."""
        ritz_values, ritz_vectors, vectors = self._extract_ritz_decomp_and_raw_base(n_ritz, k)
        return ritz_values, ritz_vectors

    def _approximate_residuals(self, n_ritz, k=None):
        # For a given eigen value/vector pair lambda_i / u_i, we have:
        #
        # ||A u_i - lambda_i u_i|| = h[k, k-1] * |<e_k, v_k>| where u_k = q * v_k
        #
        # See e.g. proposition 6.8 of
        # https://www-users.cse.umn.edu/~saad/eig_book_2ndEd.pdf
        _, _, v_k = self._extract_ritz_decomp_and_raw_base(n_ritz, k)
        return np.abs(self.h[k, k-1] * v_k[-1])

    def _residuals(self, a, n_ritz, k=None):
        ritz_values, ritz_vectors = self._extract_ritz_decomp(n_ritz, k)
        return nlin.norm(a @ ritz_vectors - ritz_values * ritz_vectors, axis=0)


def _largest_eigvals(h, n_ev):
    eigvals, eigvecs = nlin.eig(h)
    ind = np.argsort(np.abs(eigvals))[:-n_ev-1:-1]
    return eigvals[ind], eigvecs[:, ind]
