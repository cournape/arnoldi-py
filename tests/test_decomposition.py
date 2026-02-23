import numpy as np
import pytest
import scipy.sparse as sp

from arnoldi.decomposition import RitzDecomposition, arnoldi_decomposition
from arnoldi.matrices import mark, laplace
from arnoldi.utils import rand_normalized_vector

from .common import MAX_RETRIES_SHORT



ATOL = 1e-8
RTOL = 1e-4

norm = np.linalg.norm

def sort_by_criteria(x):
    idx = np.argsort(np.abs(x))
    return x[idx]


def inject_noise(A):
    A[:] = np.random.randn(*A.shape)


def basis_vector(n, k, dtype=np.int64):
    """Create the basis vector e_k in R^n, aka e_k is (n,), and with e_k[k-1] =
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
        V_m.conj().T @ V_m, np.eye(m), rtol=RTOL, atol=ATOL
    )

    # the arnoldi decomposition invariants are respected
    np.testing.assert_allclose(
        A @ V_m,
        V_m @ H_m + H[-1, -1] * np.outer(V[:, -1], e_m),
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(V_m.conj().T @ A @ V_m, H_m, rtol=RTOL, atol=ATOL)


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
        Va, Ha, n_iter = arnoldi_decomposition(A, V, H, ATOL, max_dim=max_dim)

        ## Then
        assert Va.shape == (n, max_dim+1)
        assert Ha.shape == (max_dim+1, max_dim)
        assert_invariants(A, Va, Ha, n_iter)

    def test_converge_first_iteration(self):
        ## Given
        n = 10
        m = 6
        dtype = np.complex128

        A = sp.random(n, n, density=5 / n, dtype=dtype)
        A += sp.diags_array(np.ones(n))

        ## When
        # the Arnoldi decomposition is initialized with an eigenvector
        _, r_vecs = sp.linalg.eigs(A)

        V = np.zeros((n, m+1), dtype=dtype)
        H = np.zeros((m+1, m), dtype=dtype)
        V[:, 0] = r_vecs[:, 0]

        Vm, Hm, n_iter = arnoldi_decomposition(A, V, H, ATOL)

        ## Then
        # Only one iteration is expected
        assert n_iter == 1
        assert Vm.shape == (n, n_iter+1)
        assert Hm.shape == (n_iter+1, 1)
        assert_invariants(A, Vm, Hm, n_iter)


class TestEigenValues:
    @pytest.mark.parametrize(
        "m,d", [(5, 0), (10, 1), (15, 2), (20, 3), (25, 5), (30, 7)]
    )
    @pytest.mark.flaky(reruns=MAX_RETRIES_SHORT)
    def test_mark10(self, m, d):
        # For the numerical value, see table 6.1 of Numerical Methods for Large
        # Eigenvalue Problems, 2nd edition.

        ## Given
        A = mark(10)
        n = A.shape[0]
        k = 2
        dtype = np.complex128

        ## When
        V = np.zeros((n, m+1), dtype)
        H = np.zeros((m+1, m), dtype)

        V[:, 0] = rand_normalized_vector(n, dtype)
        V, H, _ = arnoldi_decomposition(A, V, H)

        ritz = RitzDecomposition.from_v_and_h(V, H, k)

        val = ritz.values[0]
        vec = ritz.vectors[:, 0]

        ## Then
        residual = norm(A @ vec - val * vec)
        assert residual <= 2 * 10**(-d)


class TestRitzDecomposition:
    def compute_ritz(self, A, m, k, sort_function=None):
        n = A.shape[0]
        dtype = np.complex128

        V = np.zeros((n, m+1), dtype=dtype)
        H = np.zeros((m+1, m), dtype=dtype)

        V[:, 0] = rand_normalized_vector(n, dtype)

        V, H, n_iter = arnoldi_decomposition(A, V, H)
        return RitzDecomposition.from_v_and_h(V, H, k, sort_function=sort_function)

    @pytest.mark.parametrize(
        "which, sort_function", [
            ("LM", lambda x: np.argsort(-np.abs(x))),
            ("LR", lambda x: np.argsort(-np.real(x))),
        ]
    )
    @pytest.mark.flaky(reruns=MAX_RETRIES_SHORT)
    def test_simple(self, which, sort_function):
        ## Given
        A = mark(10)
        m = 30
        k = 2
        precision = 3

        r_vecs = sp.linalg.eigs(A, k, which=which)[1]

        ## When
        ritz = self.compute_ritz(A, m, k, sort_function=sort_function)

        ## Then
        overlap = np.linalg.norm(ritz.vectors.T @ r_vecs) / np.sqrt(k)
        np.testing.assert_allclose(overlap, 1, rtol=1e-4, atol=ATOL)

        residuals = norm(A @ ritz.vectors - ritz.values * ritz.vectors)
        assert residuals <= 2 * 10**(-precision)

    @pytest.mark.parametrize(
        "A,m", [(mark(10), 20), (laplace(100), 10)]
    )
    def test_residual_computation(self, A, m):
        ## Given
        k = 2

        ## When
        ritz = self.compute_ritz(A, m, k)

        ## Then
        residuals = norm(A @ ritz.vectors - ritz.values * ritz.vectors, axis=0)
        np.testing.assert_allclose(ritz.compute_true_residuals(A),
                                   residuals, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ritz.approximate_residuals, residuals,
                                   rtol=RTOL, atol=ATOL)

    def test_max_dim(self):
        # Ensure max_dim is correctly implemented in RitzDecomposition
        ## Given
        A = mark(10)
        n = A.shape[0]
        m = 20
        k = 2
        max_dim = m - 5
        dtype = np.complex128

        ## When
        V = np.zeros((n, m+1), dtype=dtype)
        H = np.zeros((m+1, m), dtype=dtype)

        V[:, 0] = rand_normalized_vector(n, dtype)
        V, H, n_iter = arnoldi_decomposition(A, V, H)

        inject_noise(V[:, max_dim:])
        inject_noise(H[max_dim+1:,max_dim:])

        broken_ritz = RitzDecomposition.from_v_and_h(V, H, k)
        ritz = RitzDecomposition.from_v_and_h(V, H, k, max_dim=max_dim)

        ## Then
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(broken_ritz.compute_true_residuals(A),
                                       broken_ritz.approximate_residuals,
                                       rtol=RTOL, atol=ATOL)

        np.testing.assert_allclose(ritz.compute_true_residuals(A),
                                   ritz.approximate_residuals, rtol=RTOL,
                                   atol=ATOL)
