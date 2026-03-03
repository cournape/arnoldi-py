import numpy as np

from .decomposition import arnoldi_decompose
from .explicit_restarts import History
from .ortho import DEFAULT_ORTHONORMALIZER
from .utils import arg_largest_magnitude, ordered_schur, rand_normalized_vector


def partial_schur(
    A, nev, *, max_dim=None, stopping_criterion=None, max_restarts=100,
    sort_function=None, p=None, orthonormalize=None,
):
    """ Compute a partial Schur decompositiokn using the Krylov-Schur algorithm

    Parameters
    ----------
    A : ndarray of shape (n, n)
        square matrix to be decomposed
    nev : int
        Number of requested eigen pairs
    """
    if stopping_criterion is None:
        tol = np.sqrt(np.finfo(A.dtype).eps)
    else:
        tol = stopping_criterion

    if sort_function is None:
        sort_function = arg_largest_magnitude

    if orthonormalize is None:
        orthonormalize = DEFAULT_ORTHONORMALIZER

    assert max_restarts > 0

    n = A.shape[0]
    assert A.shape[1] == n

    if max_dim is None:
        max_dim = min(max(2 * nev + 1, 20), n)

    # p is the size of the active size after compression. If None, use
    # "dynamic" p. In that case, we will use same logic as SLEPc:
    #   p = nconv + max(1, floor(max_dim - nconv) * keep)
    keep = 0.5
    if p is None:
        use_dynamic_p = True
    else:
        use_dynamic_p = False
        assert nev <= p < max_dim

    dtype = np.complex128

    # Using order=F significantly speeds up the cases where orthonormalization
    # is a bottleneck. Observed 3x performance increase in some cases
    V = np.zeros((n, max_dim+1), dtype=dtype, order="F")
    H = np.zeros((max_dim+1, max_dim), dtype=dtype)

    v0 = rand_normalized_vector(n, dtype)
    V[:, 0] = v0

    history = History.from_k(nev)
    has_converged = False

    V_a, H_a, m = arnoldi_decompose(
        A, V, H, max_dim=max_dim, start_dim=0, invariant_tol=tol,
        orthonormalize=orthonormalize
    )

    for restart in range(max_restarts):
        if m != max_dim:
            happy_breakdown = True
            raise ValueError("Happy breakdown not supported yet")
        else:
            happy_breakdown = False

        # FIXME: this logic is broken
        matvecs = restart * (max_dim - nev) + (m - nev)

        V_active = V_a[:, :m]
        H_active = H_a[:m, :m]

        T, Q = ordered_schur(H_active, output="complex", sort_function=sort_function)

        ## Check convergence
        approximate_residuals = np.abs(H_a[-1, -1] * Q[m-1, :])
        approximate_convergence = approximate_residuals / np.abs(np.diag(T[:, :]))

        for k in range(nev):
            if approximate_convergence[k] <= tol:
                history.matvecs[k] = matvecs
                history.restarts[k] = restart + 1

        n_converged = 0
        for k in range(nev):
            if approximate_convergence[k] > tol:
                break
            n_converged += 1

        if use_dynamic_p:
            p = n_converged + max(1, int(np.floor((max_dim - n_converged) * keep)))
        # assert to shut up the type checker
        assert p is not None

        ## Truncation
        Qp = Q[:, :p]
        Tp = T[:p, :p]

        V[:, :p] = V_active @ Qp
        # Not a typo: we copy the last vector of the non truncated orthonormal
        # basis as the last vector of the truncated basis
        V[:, p] = V[:, m]

        # Resetting H to 0 is critical in the case of dynamic p w/o locking,
        # as p may decrease between iterations  in this case. Without resetting
        # to 0, Arnoldi decomposition would use some obsolete data, breaking
        # the Arnoldi invariants.
        old_coupling = H_a[-1, :m].copy()
        H[:] = 0

        H[:p, :p] = Tp
        H[p, :p] = old_coupling @ Qp

        has_converged = happy_breakdown or n_converged >= nev
        if has_converged:
            break

        V_a, H_a, m = arnoldi_decompose(
            A, V, H, max_dim=max_dim, start_dim=p, invariant_tol=tol,
            orthonormalize=orthonormalize
        )

    if not has_converged:
        raise ValueError("Has not converged !")
    schur_vecs = V[:, :nev]
    schur_mat = H[:nev, :nev]

    # Note: returns same order as schur
    return schur_vecs, schur_mat, history
