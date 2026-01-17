import numpy as np

from scipy.linalg import schur
from scipy.linalg.lapack import ctrexc, dtrexc, strexc, ztrexc


def rand_normalized_vector(n, dtype=np.float64):
    """ Create a random normalized vector
    """
    v = np.random.randn(n).astype(dtype)
    v /= np.linalg.norm(v)

    return v


def arg_largest_magnitude(x):
    return np.argsort(-np.abs(x))


def arg_largest_real(x):
    return np.argsort(-np.real(x))


_TREXC_FUNCTION = {
    np.dtype("float32"): strexc,
    np.dtype("float64"): dtrexc,
    np.dtype("complex64"): ctrexc,
    np.dtype("complex128"): ztrexc,
}


def ordered_schur(a, output="real", *, sort_function=None):
    """Schur decomposition with eigenvalues sorted by the given sorting
    function.
    """
    if output == "complex":
        complex_dtype = np.result_type(a.dtype, 1j)
        trexc_function = _TREXC_FUNCTION[complex_dtype]
    else:
        trexc_function = _TREXC_FUNCTION[a.dtype]

    if sort_function is None:
        sort_function = arg_largest_magnitude

    T, Z = schur(a, output=output)
    n = T.shape[0]

    if output == "complex":
        eigenvalues = np.diag(T)
        ordered_indices = sort_function(eigenvalues)

        current_pos = list(range(n))

        for target, source_idx in enumerate(ordered_indices):
            source = current_pos.index(source_idx)

            if source != target:
                # + 1 is because Fortran/trexc function are 1-based indexing
                T, Z, info = trexc_function(T, Z, source + 1, target + 1)

                # Update position tracking
                moved = current_pos.pop(source)
                current_pos.insert(target, moved)
    else:
        raise ValueError("output!='complex' not implemented yet")

    return T, Z
