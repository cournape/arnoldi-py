import numpy as np

from arnoldi.krylov_schur import partial_schur
from arnoldi.matrices import mark
from arnoldi.utils import arg_largest_real


norm = np.linalg.norm


class TestPartialSchur:
    def test_mark10(self):
        ## Given
        A = mark(10)
        m = 5
        k = 3
        max_restarts = 1000

        ## When
        Q, T, _ = partial_schur(
            A, k, max_dim=m, sort_function=arg_largest_real, max_restarts=max_restarts
        )

        ## Then
        residuals = norm(A @ Q - Q @ T, axis=1)
        np.testing.assert_allclose(residuals, 0, rtol=1e-4, atol=1e-08)

    def test_simple_diagonal(self):
        # Test a simple orthonormally projected diagonal matrix

        ## Given
        D = np.diag([7, 7, 5, 4, 3, 2, 1])
        n = D.shape[0]
        M = np.random.randn(n, n)
        Q, _ = np.linalg.qr(M)
        A = Q.T @ D @ Q

        k = 3
        m = n - 1
        max_restarts = 1000

        ## When
        Q, T, _ = partial_schur(
            A, k, max_dim=m, sort_function=arg_largest_real, max_restarts=max_restarts
        )

        ## Then
        residuals = norm(A @ Q - Q @ T, axis=1)
        np.testing.assert_allclose(residuals, 0, rtol=1e-4, atol=1e-08)
