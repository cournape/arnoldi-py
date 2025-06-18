import numpy as np
import pytest
import scipy.sparse as sp

from arnoldi import Arnoldi


ATOL = 1e-8
RTOL = 1e-4


def largest_eigvals(m, k):
    """ Compute top k eigenvalues of m, when sorted by amplitude.
    """
    r_eigvals, _ = np.linalg.eig(m)
    return np.array(sorted(r_eigvals, key=np.abs, reverse=True)[:k])


class TestArnoldiExpansion:
    def test_invariant_simple(self):
        ## Test the invariant A * Q ~ Q * H, with H Hessenberg matrix and Q
        ## orthonormal

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

    @pytest.mark.flaky(reruns=3)
    def test_eigvals_simple(self):
        # Given
        n = 20
        ## We use large krylov space since basic arnoldi does not converge
        ## quickly. Once we implement restarts, it should converge much faster
        k = n - 1

        a = sp.random(n, n, density=5 / n, dtype=np.complex128)
        # We add ones on the diag to have a well conditioned matrix and eigen
        # values not too far from 1
        a += sp.diags_array(np.ones(n))

        # When
        arnoldi = Arnoldi(n, k)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(a)

        q, h = arnoldi.q[:, :n_iter+1], arnoldi.h[:n_iter, :n_iter]

        r_eigvals = largest_eigvals(a.toarray(), 3)
        ritz_values = largest_eigvals(h, 3)

        # Then
        np.testing.assert_allclose(r_eigvals, ritz_values, rtol=RTOL, atol=ATOL)
