import pytest

import numpy as np

from arnoldi.utils import ordered_schur


# Values taken from scipy.sparse.linalg ARPACK tests
def _get_test_tolerance(type_char):
    rtol = {
        "f": 3000 * np.finfo(np.float32).eps,
        "d": 2000 * np.finfo(np.float64).eps,
    }
    for k in ["f", "d"]:
        rtol[k.upper()] = rtol[k]

    atol = rtol

    return rtol[type_char], atol[type_char]


class TestOrderedSchur:
    @pytest.mark.parametrize("dtype", ["F", "D"])
    def test_simple_complex(self, dtype):
        ## Given
        r_T = np.array([
            [5.0, 1.5, 0.8, 0.1, 0.4],
            [0.0, 4.0, 1.2, 1.0, 0.5],
            [0.0, 0.0, 3.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 2.0, 0.6],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]).astype(dtype)

        r_Q, _ = np.linalg.qr(np.random.randn(*r_T.shape).astype(dtype))
        A = r_Q.T @ r_T @ r_Q

        rtol, atol = _get_test_tolerance(dtype)

        ## When
        T, Q = ordered_schur(
            A, output="complex", sort_function=lambda v: np.argsort(v)
        )

        ## Then
        assert T.dtype == np.dtype(dtype)
        assert Q.dtype == np.dtype(dtype)

        np.testing.assert_allclose(Q @ T @ Q.T.conj(), A, rtol=rtol, atol=atol)
        np.testing.assert_allclose(np.diag(T), [1, 2, 3, 4, 5], rtol=rtol, atol=atol)

    @pytest.mark.xfail(reason="real mode not implemented yet")
    @pytest.mark.parametrize("dtype", ["f", "d"])
    def test_simple_real(self, dtype):
        ## Given
        r_T = np.array([
            [1.0, 1.5, 0.8, 0.1, 0.4],
            [0.0, 2.0, 1.2, 1.0, 0.5],
            [0.0,-0.3, 2.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0,-2.0, 4.0],
        ]).astype(dtype)

        complex_dtype = np.result_type(dtype, 1j)

        r_eivals = np.array([
            4 + 1j * np.sqrt(2),
            4 - 1j * np.sqrt(2),
            2 + 1j * np.sqrt(1.2 * 0.3),
            2 - 1j * np.sqrt(1.2 * 0.3),
            1,
        ]).astype(complex_dtype)

        r_Q, _ = np.linalg.qr(np.random.randn(*r_T.shape).astype(dtype))
        A = r_Q.T @ r_T @ r_Q

        rtol, atol = _get_test_tolerance(dtype)

        ## When
        T, Q = ordered_schur(
            A, output="real", sort_function=lambda v: np.argsort(-np.abs(v))
        )

        ## Then
        assert T.dtype == np.dtype(dtype)
        assert Q.dtype == np.dtype(dtype)

        np.testing.assert_allclose(Q @ T @ Q.T.conj(), A, rtol=rtol, atol=atol)
        np.testing.assert_allclose(np.linalg.eigvals(T), r_eivals, rtol=rtol, atol=atol)
