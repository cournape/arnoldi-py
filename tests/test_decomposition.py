import numpy as np
import scipy.sparse as sp

from arnoldi import Arnoldi


def largest_eigvals(m, k):
    """ Compute top k eigenvalues of m, when sorted by amplitude.
    """
    r_eigvals, _ = np.linalg.eig(m)
    return np.array(sorted(r_eigvals, key=np.abs, reverse=True)[:k])


RTOL = 1e-5
ATOL= 1e-8

class TestArnoldiSimple:
    def test_simple(self):
        # Given
        n = 10
        k = 6

        a = sp.random(n, n, density=5 / n, dtype=np.complex128)
        a += sp.diags_array(np.ones(n))

        # When
        arnoldi = Arnoldi(n, k)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(a)

        q, h = arnoldi.q[:, :n_iter+1], arnoldi.h[:n_iter+1, :n_iter+1]

        # Then
        ## the Q arnoldi basis is orthonormal
        np.testing.assert_allclose(
            q.conj().T @ q, np.eye(n_iter+1), rtol=RTOL, atol=ATOL
        )
        ## the arnoldi decomposition invariant is respected
        remain = a @ q[:, :n_iter] - q @ h
        np.testing.assert_allclose(
            a @ q[:, :n_iter], q @ h, rtol=RTOL, atol=ATOL
        )

    def test_eigvals(self):
        # Given
        n = 20
        ## We use large krylov space since basic arnoldi does not converge
        ## quickly. Once we implement restarts, it should converge much faster
        k = n - 1

        a = sp.random(n, n, density=5 / n, dtype=np.complex128)
        a += sp.diags_array(np.ones(n))

        # When
        arnoldi = Arnoldi(n, k)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(a)

        q, h = arnoldi.q[:, :n_iter+1], arnoldi.h[:n_iter+1, :n_iter+1]

        r_eigvals = largest_eigvals(a.toarray(), 3)
        ritz_values = largest_eigvals(h[:-1, :], 3)

        # Then
        np.testing.assert_allclose(
            r_eigvals, ritz_values, rtol=1000 * RTOL, atol=ATOL
        )
