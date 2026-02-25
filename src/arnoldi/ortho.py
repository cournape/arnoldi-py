import numpy as np
from scipy.linalg import get_blas_funcs

nrm2, gemv = get_blas_funcs(("nrm2", "gemv"), dtype=np.complex128)

M_SQRT1_2 = np.sqrt(0.5)


def dgks_mgs(w: np.ndarray, V: np.ndarray, h: np.ndarray, tol: float=1e-8,
             eta=M_SQRT1_2):
    """ Reorthonormalization using Modified Gram-Schmidt, with DGKS-controlled
    double re-orthonormalization

    Parameters
    ----------
    w: ndarray of shape (n,)
        The array to orthonormalize
    V: ndarray of shape (n, j)
        The basis to orthonormalize against
    h: ndarray of shape (j,)
        The array to accumulate the scalar products (modified in place)
    eta: float
        Double reorthonormalization will be done if norm(w after ortho) /
        norm(w before ortho) < eta

    Returns
    -------
    beta: float
        the final norm of w after orthonormalization
    breakdown: bool
        True if w could not be orthonormalized against V, i.e. when w is in the
        span of V
    """
    j = V.shape[1]

    before = np.linalg.norm(w)

    # Modified Gram-Schmidt (MGS) for orthonormalization
    for i in range(j):
        h[i] = np.vdot(V[:, i], w)
        w -= h[i] * V[:, i]

    after = np.linalg.norm(w)

    # DGKS criterian for double MGS
    if after < eta * before:
        for i in range(j):
            coeff = np.vdot(V[:, i], w)
            h[i] += coeff
            w -= coeff * V[:, i]

    beta = np.linalg.norm(w)
    return beta, beta < tol


def dgks_gs(w: np.ndarray, V: np.ndarray, h: np.ndarray, tol: float=1e-8,
            eta=M_SQRT1_2):
    """ Double reorthonormalization using Gram-Schmidt, with DGKS criterion to
    trigger double orthonormalization.

    Parameters
    ----------
    w: ndarray of shape (n,)
        The array to orthonormalize
    V: ndarray of shape (n, j)
        The basis to orthonormalize against
    h: ndarray of shape (j,)
        The array to accumulate the scalar products (modified in place)
    eta: float
        Double reorthonormalization will be done if norm(w after ortho) /
        norm(w before ortho) < eta

    Returns
    -------
    beta: float
        the final norm of w after orthonormalization
    breakdown: bool
        True if w could not be orthonormalized against V, i.e. when w is in the
        span of V

    Notes
    -----

    Using DGKS criterion for double orthonormalization with standard Gram
    Schmidt (GS) is enough, and modified GS superfluous. In Python, GS can
    be implemented without a loop so will be significantly faster. See "On
    the loss of orthogonality in the Gram-Schmidt orthogonalization process",
    from Giraud, Langou, Rozložník (2005) for details.
    """
    j = V.shape[1]

    beta_before = nrm2(w)

    tmp = gemv(1.0, V, w, trans=2)
    h[:j+1] = tmp
    w -= gemv(1.0, V, tmp)

    beta = nrm2(w)

    # DGKS criterian for double MGS
    if beta < beta_before * eta:
        tmp = gemv(1.0, V, w, trans=2)
        h[:j+1] += tmp
        w -= gemv(1.0, V, tmp)
        beta = nrm2(w)

    return beta, beta < tol
