import numpy as np
import scipy.sparse as sp


class Arnoldi:
    """ Create an arnoldi solver for operators of dimension n, with a Krylov
    basis of k dimensions.
    """
    def __init__(self, n, k, dtype=np.complex128):
        self.n = n
        self.k = k

        self.q = np.zeros((n, k+1), dtype=dtype)
        self.h = np.zeros((k+1, k), dtype=dtype)

        self.dtype = dtype

    def initialize(self, init_vector=None):
        if init_vector is None:
            init_vector = np.random.randn(self.n).astype(self.dtype)
            init_vector /= np.linalg.norm(init_vector)

        self.q[:, 0] = init_vector

    def iterate(self, a, tol=1e-12):
        for j in range(self.k):
            v = self.q[:, j+1]
            v[:] = a @ self.q[:, j]

            # Modified Gram-Schmidt (orthonormalization)
            for i in range(j + 1):
                self.h[i, j] = np.vdot(self.q[:, i], v)
                v -= self.h[i, j] * self.q[:, i]

            self.h[j + 1, j] = np.linalg.norm(v)

            if self.h[j + 1, j] < tol:
                return j
            v /= self.h[j + 1, j]

        return self.k
