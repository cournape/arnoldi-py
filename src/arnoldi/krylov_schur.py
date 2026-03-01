import numpy as np

from .decomposition import arnoldi_decomposition
from .explicit_restarts import History
from .utils import arg_largest_magnitude, ordered_schur, rand_normalized_vector


def partial_schur(
    A, nev, *, max_dim=None, stopping_criterion=None, max_restarts=100,
    sort_function=None, p=None,
):
    """ Compute a partial Schur decompositiokn using the Krylov-Schur algorithm
    """
    if stopping_criterion is None:
        tol = np.sqrt(np.finfo(A.dtype).eps)
    else:
        tol = stopping_criterion

    if sort_function is None:
        sort_function = arg_largest_magnitude

    assert max_restarts > 0

    n = A.shape[0]
    assert A.shape[1] == n

    if max_dim is None:
        max_dim = min(max(2 * nev + 1, 20), n)

    # p is the size of the active size after compression. If None, use
    # "dynamic" p. In that case, we will use same logic as SLEPc:
    #   p = nconv + max(1, floor(max_dim - nconv) * keep)
    p = None
    keep = 0.5
    if p is None:
        use_dynamic_p = True
    else:
        use_dynamic_p = False
        assert nev <= p < max_dim

    assert p is None
    dtype = np.complex128

    # Using order=F significantly speeds up the cases where orthonormalization
    # is a bottleneck. Observed 3x performance increase in some cases
    V = np.zeros((n, max_dim+1), dtype=dtype, order="F")
    H = np.zeros((max_dim+1, max_dim), dtype=dtype)

    v0 = rand_normalized_vector(n, dtype)
    V[:, 0] = v0

    history = History.from_k(nev)
    has_converged = False

    V_a, H_a, n_iter = arnoldi_decomposition(
        A, V, H, max_dim=max_dim, start_dim=0, invariant_tol=tol
    )
    m = n_iter

    for restart in range(max_restarts):
        if m != max_dim:
            happy_breakdown = True
            raise ValueError("Happy breakdown not supported yet")
        else:
            happy_breakdown = False

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

        if restart % 100 == 0:
            print(f"  it={restart:3d} nconv={n_converged:3d} nev={nev:3d} p={p:3d} "
                f"next Arnoldi: [{p} .. {max_dim}["
            )

        ## Truncation
        Qp = Q[:, :p]
        Tp = T[:p, :p]

        V[:, :p] = V_active @ Qp
        # Not a typo: we copy the last vector of the non truncated orthonormal
        # basis as the last vector of the truncated basis
        V[:, p] = V[:, m]

        H[:p, :p] = Tp
        # FIXME: there is a simplification possible as all entries of H_a[m, :]
        # except one are supposed to be 0
        old_coupling = H_a[-1, :m]
        H[p, :p] = old_coupling @ Qp
        H[p, p:] = 0 # Should be unecessary as those entries are not used in the next Arnoldi expansion

        # Critical in the case p decreased, which may happen with dynamic p
        # logic (p=None argument). We reset the old values as those are
        # garbaged now
        H[p+1:, :p] = 0

        has_converged = happy_breakdown or n_converged >= nev
        if has_converged:
            break

        ### Testing invariants
        #
        ## V[:, :p+1] is orthonormal
        #G = V[:, :p+1].conj().T @ V[:, :p+1]
        #assert np.linalg.norm(G - np.eye(p+1)) < tol
        #
        ## Krylov decomposition is true
        #b = H[p, :p]
        #R = A @ V[:, :p] - V[:, :p] @ H[:p, :p] - np.outer(V[:, p], b)
        #if np.linalg.norm(R) > 2 * tol:
        #    print(f"WARNING: iter={restart:4d} -> Krylov decomp res is {np.linalg.norm(R):.4g}")

        V_a, H_a, n_iter = arnoldi_decomposition(
            A, V, H, max_dim=max_dim, start_dim=p, invariant_tol=tol
        )
        m = n_iter

    if not has_converged:
        raise ValueError("Has not converged !")
    schur_vecs = V[:, :nev]
    schur_mat = H[:nev, :nev]

    # Note: returns same order as schur
    return schur_vecs, schur_mat, history
