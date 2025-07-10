import numpy as np
import numpy.linalg as nlin
import pytest
import scipy.sparse as sp

from arnoldi import ArnoldiDecomposition
from arnoldi.decomposition import _largest_eigvals, RitzDecomposition


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
        max_dim = 6
        dtype = np.complex128
        e_k = basis_vector(max_dim, max_dim, dtype)

        a = sp.random(n, n, density=5 / n, dtype=dtype)
        a += sp.diags_array(np.ones(n))

        ## When
        arnoldi = ArnoldiDecomposition(n, max_dim)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(a)

        v, h = arnoldi.v, arnoldi.h
        v_k, h_k = arnoldi._extract_arnold_decomp(n_iter)

        ## Then
        # the arnoldi basis V is orthonormal
        np.testing.assert_allclose(
            v.conj().T @ v, np.eye(n_iter + 1), rtol=RTOL, atol=ATOL
        )
        # the arnoldi decomposition invariants are respected
        np.testing.assert_allclose(
            a @ v_k,
            v_k @ h_k + h[-1, -1] * np.outer(v[:, -1], e_k),
            rtol=RTOL,
            atol=ATOL,
        )

        np.testing.assert_allclose(a @ v[:, :-1], v @ h, rtol=RTOL, atol=ATOL)

    @pytest.mark.flaky(reruns=3)
    def test_eigvals_simple(self):
        # Simple test checking that eigen values of H_k (aka the ritz values)
        # approximate the matrix's eigenvalues.

        ## Given
        n = 20
        # We use large krylov space since basic arnoldi does not converge
        # quickly. Once we implement restarts, it should converge much faster
        k = n - 1
        n_ev = 3

        a = sp.random(n, n, density=5 / n, dtype=np.complex128)
        # We add ones on the diag to have a well conditioned matrix and eigen
        # values not too far from 1
        a += sp.diags_array(np.ones(n))

        r_eigvals, _ = _largest_eigvals(a.toarray(), n_ev)

        ## When
        arnoldi = ArnoldiDecomposition(n, k)
        arnoldi.initialize()
        arnoldi.iterate(a)

        ritz = RitzDecomposition.from_arnoldi(arnoldi, n_ev)

        ## Then
        # Ensure estimated eigen values approximately match
        np.testing.assert_allclose(r_eigvals, ritz.values, rtol=RTOL, atol=ATOL)

    def test_residuals_computation(self):
        ## Given
        n = 20
        k = 6
        n_ev = 3

        a = sp.random(n, n, density=5 / n, dtype=np.complex128)
        # We add ones on the diag to have a well conditioned matrix and eigen
        # values not too far from 1
        a += sp.diags_array(np.ones(n))

        ## When
        arnoldi = ArnoldiDecomposition(n, k)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(a)

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
            ritz = RitzDecomposition.from_arnoldi(arnoldi, n_ev, i)
            r_residuals = nlin.norm(a @ ritz.vectors - ritz.values * ritz.vectors, axis=0)
            np.testing.assert_allclose(r_residuals, ritz.approximate_residuals, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(r_residuals, ritz.compute_residuals(a))
