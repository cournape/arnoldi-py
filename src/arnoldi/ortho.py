import numpy as np


def double_mgs(w: np.ndarray, V: np.ndarray, h: np.ndarray, tol: float=1e-8):
    """ Double reorthonormalization using Modified Gram-Schmidt

    Parameters
    ----------
    w: ndarray of shape (n,)
        The array to orthonormalize
    V: ndarray of shape (n, j)
        The basis to orthonormalize against
    h: ndarray of shape (j,)
        The array to accumulate the scalar products (modified in place)

    Returns
    -------
    beta: float
        the final norm of w after orthonormalization
    breakdown: bool
        True if w could not be orthonormalized against V, i.e. when w is in the
        span of V
    """
    j = V.shape[1] - 1

    # Modified Gram-Schmidt (MGS) for orthonormalization
    for i in range(j + 1):
        h[i] = np.vdot(V[:, i], w)
        w -= h[i] * V[:, i]

    # Double reorthonormalization with GMS
    for i in range(j + 1):
        coeff = np.vdot(V[:, i], w)
        h[i] += coeff
        w -= coeff * V[:, i]

    beta = np.linalg.norm(w)
    return beta, beta < tol
