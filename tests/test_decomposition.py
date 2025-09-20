import numpy as np
import pytest
import scipy.sparse as sp

from arnoldi import Arnoldi
from arnoldi.decomposition import _largest_eigvals


ATOL = 1e-8
RTOL = 1e-4


def basis_vector(n, k, dtype=np.int64):
    """Create the basis vector e_k in R^n, aka e_k is (n,), and with e_k[k] =
    1
    """
    ret = np.zeros(n, dtype=dtype)
    ret[k - 1] = 1
    return ret


class TestArnoldiExpansion:
    def test_invariant_simple(self):
        # Test the invariant A * V ~ V * H, with H Hessenberg matrix and V
        # orthonormal

        ## Given
        n = 10
        m = 6
        dtype = np.complex128
        e_k = basis_vector(m, m, dtype)

        A = sp.random(n, n, density=5 / n, dtype=dtype)
        A += sp.diags_array(np.ones(n))

        ## When
        arnoldi = Arnoldi(n, m)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(A)

        V, H = arnoldi.V, arnoldi.H
        V_m, H_m = arnoldi._extract_arnold_decomp(n_iter)

        ## Then
        # the arnoldi basis V is orthonormal
        np.testing.assert_allclose(
            V.conj().T @ V, np.eye(n_iter + 1), rtol=RTOL, atol=ATOL
        )
        # the arnoldi decomposition invariants are respected
        np.testing.assert_allclose(
            A @ V_m,
            V_m @ H_m + H[-1, -1] * np.outer(V[:, -1], e_k),
            rtol=RTOL,
            atol=ATOL,
        )

        np.testing.assert_allclose(A @ V[:, :-1], V @ H, rtol=RTOL, atol=ATOL)

    @pytest.mark.flaky(reruns=3)
    def test_eigvals_simple(self):
        # Simple test checking that eigen values of H_k (aka the ritz values)
        # approximate the matrix's eigenvalues.

        ## Given
        n = 20
        # We use large krylov space since basic arnoldi does not converge
        # quickly. Once we implement restarts, it should converge much faster
        m = n - 1
        n_ev = 3

        A = sp.random(n, n, density=5 / n, dtype=np.complex128)
        # We add ones on the diag to have a well conditioned matrix and eigen
        # values not too far from 1
        A += sp.diags_array(np.ones(n))

        r_eigvals, _ = _largest_eigvals(A.toarray(), n_ev)

        ## When
        arnoldi = Arnoldi(n, m)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(A)

        ritz_values, _ = arnoldi._extract_ritz_decomp(n_ev, n_iter)

        ## Then
        # Ensure estimated eigen values approximately match
        np.testing.assert_allclose(r_eigvals, ritz_values, rtol=RTOL, atol=ATOL)

    def test_residuals_computation(self):
        ## Given
        n = 20
        m = 6
        n_ev = 3

        A = sp.random(n, n, density=5 / n, dtype=np.complex128)
        # We add ones on the diag to have a well conditioned matrix and eigen
        # values not too far from 1
        A += sp.diags_array(np.ones(n))

        ## When
        arnoldi = Arnoldi(n, m)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(A)

        ## Then
        # Ensure residuals and approximate residuals match

        # For a given ritz value/vector pair lambda_i / u_i, we have, in precise
        # arithmetic:
        #
        #   ||A u_i - lambda_i u_i|| = h[k, k-1] * |<e_k, v_k>| where u_k = q * v_k
        #
        # In practice, they only approximately equal. The left hand side is the
        # "true" residual, the right hand side the approximate one
        for i in range(1, n_iter):
            residuals = arnoldi._approximate_residuals(n_ev, i)
            r_residuals = arnoldi._residuals(A, n_ev, i)
            np.testing.assert_allclose(r_residuals, residuals, rtol=RTOL, atol=ATOL)
