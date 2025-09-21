import numpy as np
import pytest
import scipy.sparse as sp

from arnoldi import ArnoldiDecomposition
from arnoldi.decomposition import _largest_eigvals, arnoldi_decomposition
from arnoldi.matrices import mark
from arnoldi.utils import rand_normalized_vector


ATOL = 1e-8
RTOL = 1e-4

norm = np.linalg.norm


def basis_vector(n, k, dtype=np.int64):
    """Create the basis vector e_k in R^n, aka e_k is (n,), and with e_k[k] =
    1
    """
    ret = np.zeros(n, dtype=dtype)
    ret[k - 1] = 1
    return ret


def assert_invariants(A, V, H, m):
    """Raise an assertion error if Arnoldi decomposition invariants are not
    respected.

    The key Arnoldi invariant are
    1. V should be orthonormal
    2. A * V_m ~ V_m * H_m + h_{m+1,m} v_{m+1}
    3. V_m` * A * V_m ~ H_m, with H_m Hessenberg matrix

    Reference
    ---------
    See eqs. 6.8 and 6.9 of Numerical Methods for Large Eigenvalue
    Problems, 2nd edition.
    """
    e_m = basis_vector(m, m, V.dtype)

    V_m = V[:, :m]
    H_m = H[:m, :m]

    # the arnoldi basis V is orthonormal
    np.testing.assert_allclose(
        V.conj().T @ V, np.eye(m + 1), rtol=RTOL, atol=ATOL
    )

    # the arnoldi decomposition invariants are respected
    np.testing.assert_allclose(
        A @ V_m,
        V_m @ H_m + H[-1, -1] * np.outer(V[:, -1], e_m),
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(V_m.conj().T @ A @ V_m, H_m, rtol=RTOL, atol=ATOL)


class TestArnoldiDecomposition:
    def test_invariant_simple(self):
        # Test the invariant A * V ~ V * H, with H Hessenberg matrix and V
        # orthonormal

        ## Given
        n = 10
        m = 6
        dtype = np.complex128

        A = sp.random(n, n, density=5 / n, dtype=dtype)
        A += sp.diags_array(np.ones(n))

        ## When
        arnoldi = ArnoldiDecomposition(n, m)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(A)

        V, H = arnoldi.V, arnoldi.H

        ## Then
        # the arnoldi invariants are respected
        assert_invariants(A, V, H, n_iter)

    @pytest.mark.flaky(reruns=5)
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
        arnoldi = ArnoldiDecomposition(n, m)
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
        n_ev = 2

        A = sp.random(n, n, density=5 / n, dtype=np.complex128)
        # We add ones on the diag to have a well conditioned matrix and eigen
        # values not too far from 1
        A += sp.diags_array(np.ones(n))

        ## When
        arnoldi = ArnoldiDecomposition(n, m)
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


class TestArnoldiDecompositionFunction:
    def test_invariant_simple(self):
        ## Given
        n = 10
        m = 6
        dtype = np.complex128

        A = sp.random(n, n, density=5 / n, dtype=dtype)
        A += sp.diags_array(np.ones(n))

        V = np.zeros((n, m+1), dtype=dtype)
        H = np.zeros((m+1, m), dtype=dtype)

        V[:, 0] = rand_normalized_vector(n, dtype)

        ## When
        Va, Ha, n_iter = arnoldi_decomposition(A, V, H, ATOL)

        ## Then
        assert_invariants(A, Va, Ha, n_iter)

    def test_max_dim_support(self):
        ## Given
        n = 10
        m = 6
        max_dim = 3
        dtype = np.complex128

        A = sp.random(n, n, density=5 / n, dtype=dtype)
        A += sp.diags_array(np.ones(n))

        V = np.zeros((n, m+1), dtype=dtype)
        H = np.zeros((m+1, m), dtype=dtype)

        V[:, 0] = rand_normalized_vector(n, dtype)

        ## When
        Va, Ha, n_iter = arnoldi_decomposition(A, V, H, ATOL, max_dim)

        ## Then
        assert_invariants(A, Va, Ha, n_iter)


class TestEigenValues:
    @pytest.mark.parametrize(
        "m,d", [(5, 0), (10, 1), (15, 2), (20, 3), (25, 5), (30, 7)]
    )
    @pytest.mark.flaky(reruns=3)
    def test_mark10(self, m, d):
        # For the numerical value, see table 6.1 of Numerical Methods for Large
        # Eigenvalue Problems, 2nd edition.

        ## Given
        A = mark(10)
        n = A.shape[0]
        k = 1

        r_val = sp.linalg.eigs(A, k)[0]

        ## When
        arnoldi = ArnoldiDecomposition(n, m)
        arnoldi.initialize()
        n_iter = arnoldi.iterate(A)

        V_m, H_m = arnoldi._extract_arnoldi_decomp(n_iter)

        ritz_values, vectors = _largest_eigvals(H_m, k)
        ritz_vectors = V_m @ vectors
        val = ritz_values[0]
        vec = ritz_vectors[:, 0]

        ## Then
        residual = norm(A @ vec - val * vec)
        assert residual <= 2 * 10**(-d)
