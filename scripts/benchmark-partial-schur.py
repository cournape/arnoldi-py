import argparse
import os.path
import sys
import time

import numpy as np

from scipy.sparse.linalg import eigs

from arnoldi import partial_schur

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from utils import (
    WHICH_TO_SORT, MatvecCounter, find_best_matching, load_suitesparse_mat
)


NEV = 6
NVC = 20
TOL = 1e-8
WHICH = "LR"
MAX_RESTARTS = 40_000
DTYPE = np.complex128


def run_partial_schur(A, nev, nvc, which):
    A = MatvecCounter(A)
    t0 = time.perf_counter()

    Q, T, history = partial_schur(
        A,
        nev,
        max_dim=nvc,
        stopping_criterion=TOL,
        max_restarts=MAX_RESTARTS,
        sort_function=WHICH_TO_SORT[which],
    )
    elapsed = time.perf_counter() - t0

    vals, S = np.linalg.eig(T)
    vecs = Q @ S

    n_iters = np.max(history.restarts)

    return vals, vecs, A.matvecs, n_iters, elapsed


def run_arpack(A, nev, nvc, which):
    A = MatvecCounter(A)

    t0 = time.perf_counter()
    vals, vecs = eigs(
        A,
        k=nev,
        which=which,
        ncv=nvc,
        tol=TOL,
        maxiter=MAX_RESTARTS,
    )
    elapsed = time.perf_counter() - t0

    matvecs = A.matvecs
    # Approximation of the Arnoldi method
    n_iters = (matvecs - nvc) // (nvc - nev)

    idx = WHICH_TO_SORT[which](vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs, matvecs, n_iters, elapsed


def main():
    path = sys.argv[1]

    A = load_suitesparse_mat(path)
    A = A.astype(DTYPE)

    n = A.shape[0]
    nnz = A.nnz

    if NVC is None:
        nvc = min(max(2 * NEV + 1, 20), 20)
    else:
        nvc = NVC

    print(f"Matrix: {path}")
    print(f"  shape={n}x{n}, nnz={nnz}, dtype={A.dtype}")
    print(
        f"  nev={NEV}, tol={TOL}, nvc={nvc}, "
        f"max_restarts={MAX_RESTARTS}, which={WHICH}"
    )

    vals, vecs, matvecs, n_iters, elapsed = run_partial_schur(A, NEV, nvc,
                                                              WHICH)
    residuals = np.linalg.norm(A @ vecs - vals * vecs, axis=0)
    norm_residuals = residuals / np.abs(vals)

    assert np.all(norm_residuals < TOL * 5)

    print(
        f"  KRYLOV SCHUR: matvecs={matvecs}, elapsed={elapsed:.2f}s for {n_iters} iterations"
    )
    r_vals, r_vecs, r_matvecs, r_n_iters, r_elapsed = run_arpack(A, NEV, nvc, WHICH)
    print(
        f"  ARPACK:       matvecs={r_matvecs}, elapsed={r_elapsed:.2f}s for {r_n_iters} iterations"
    )

    print(
        f"  KRYLOV SCHUR: {1000 * elapsed/matvecs:.2f} ms per matvec {1000 * elapsed/n_iters:.2g} ms per iter"
    )
    print(
        f"  ARPACK:       {1000 * r_elapsed/r_matvecs:.2f} ms per matvec {1000 * r_elapsed/r_n_iters:.2g} ms per iter"
    )

    print(r_vals)
    print(vals)

    # Ensure the eigenvalues match. This check + ensure normalized residuals
    # are close to 0 should be enough to ensure the output is correct
    x, y = find_best_matching(r_vals, vals)
    np.testing.assert_allclose(x, y, rtol=TOL)


if __name__ == "__main__":
    main()
