import pytest

from arnoldi.explicit_restarts import naive_explicit_restarts
from arnoldi.matrices import mark

from .common import MAX_RETRIES_SHORT


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
