import pytest

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigs

from arnoldi.explicit_restarts import (
    naive_explicit_restarts,
    explicit_restarts_with_deflation,
)
from arnoldi.matrices import mark
from arnoldi.utils import arg_largest_real

from .common import MAX_RETRIES_SHORT


norm = np.linalg.norm


def find_best_matching(a, b):
    """
    Reorder both arrays so that they match as closely as possible

    Example:
    >>> x = [0, 2, 1]
    >>> y = [2, 1, 3]
    >>> find_best_matching(x, y)
    np.array([0, 1, 2]), np.array([3, 1, 2])
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Compute pairwise distances
    # Cost matrix: |a[i] - b[j]| for all pairs
    cost_matrix = np.abs(a[:, np.newaxis] - b[np.newaxis, :])

    # Find optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Check that matched pairs are close
    return a[row_ind], b[col_ind]


class TestNaiveExplicitRestarts:
    @pytest.mark.parametrize(
        "restarts, digits", [(1, 0), (2, 1), (3, 3), (4, 5), (5, 6)]
    )
    @pytest.mark.flaky(reruns=MAX_RETRIES_SHORT)
    def test_mark10(self, restarts, digits):
        # For the numerical value, see table 6.2 of Numerical Methods for Large
        # Eigenvalue Problems, 2nd edition.

        ## Given
        A = mark(10)
        m = 10

        ## When
        ritz, *_ = naive_explicit_restarts(A, m, max_restarts=restarts)

        ## Then
        assert ritz.compute_true_residuals(A) <= 2 * 10**(-digits)

    @pytest.mark.flaky(reruns=MAX_RETRIES_SHORT)
    def test_convergence(self):
        ## Given
        A = mark(10)
        m = 20
        atol = 1e-6

        ## When
        ritz, has_converged, *_ = naive_explicit_restarts(A, m,
                                                          max_restarts=200,
                                                          stopping_criterion=atol)

        ## Then
        assert ritz.compute_true_residuals(A) <= atol
        assert has_converged


def sort_by_real(x):
    return np.argsort(-np.real(x))


class TestExplicitRestartsWithDeflation:
    def ensure_values_match_with_arpack(self, A, k, max_dim=None, which="LM", tol=None, max_restarts=100):
        match which:
            case "LM":
                # To ensure default handling path is tested
                sort_function = None
            case "LR":
                sort_function = arg_largest_real
            case _:
                raise ValueError(f"Mode {which} not supported")

        r_vals = eigs(A, k, which=which)[0]

        ## When
        vals, vecs, history = explicit_restarts_with_deflation(
            A, k, max_dim=max_dim, stopping_criterion=tol, sort_function=sort_function,
            max_restarts=max_restarts,
        )
        residuals = norm(A @ vecs - vals * vecs, axis=0)

        ## Then
        assert history.k == k
        np.testing.assert_allclose(residuals, 0, rtol=1e-4, atol=1e-08)
        # Test we get the right eigenvalues according to sorting criteria
        # Note: ARPACK order for eigenvalues is undefined
        vals, r_vals = find_best_matching(vals, r_vals)
        np.testing.assert_allclose(vals, r_vals, rtol=1e-4, atol=1e-08)


    @pytest.mark.flaky(reruns=MAX_RETRIES_SHORT)
    def test_mark10(self):
        # For the numerical value, see table 6.3 of Numerical Methods for Large
        # Eigenvalue Problems, 2nd edition.

        ## Given
        A = mark(10)
        m = 10
        k = 3
        tol = 1e-8

        ## When / then
        self.ensure_values_match_with_arpack(A, k, m, which="LR", tol=tol)

    def test_simple(self):
        # Test a simple orthonormally projected diagonal matrix

        ## Given
        D = np.diag([7, 7, 5, 4, 3, 2, 1])
        n = D.shape[0]
        M = np.random.randn(n, n)
        Q, _ = np.linalg.qr(M)
        A = Q.T @ D @ Q

        k = 3

        ## When / then
        self.ensure_values_match_with_arpack(A, k)

    def test_fail_convergence(self):
        # Test a simple orthonormally projected diagonal matrix

        ## Given
        A = mark(10)
        k = 3
        # Very low tolerance, but few restarts and small Krylov space dimension
        # to cause convergence failures
        max_restarts = 10
        tol = 1e-16
        max_dim = 5

        ## When / then
        with pytest.raises(ValueError, match="Could not converge for value 0"):
            self.ensure_values_match_with_arpack(
                A, k, max_dim=max_dim, tol=tol, max_restarts=max_restarts
            )
