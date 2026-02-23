# AI note: this script was mostly generated using AI (claude code)
import argparse
import os.path
import sys
import time

import numpy as np

import scipy.io
import scipy.sparse as sp

from scipy.linalg import toeplitz
from scipy.sparse.linalg import eigs, LinearOperator

from arnoldi.krylov_schur import partial_schur
from arnoldi.utils import arg_largest_magnitude, arg_largest_real


HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from utils import (
    WHICH_TO_SORT, MatvecCounter, load_suitesparse_mat, print_residuals
)


def grcar_matrix(n, k=3):
    """Grcar matrix - nonsymmetric Toeplitz with known spectral properties"""
    c = np.zeros(n)
    c[0] = 1
    r = np.zeros(n)
    r[: min(k + 1, n)] = 1
    return toeplitz(c, r)


def clement_matrix(n):
    """Tridiagonal matrix with known eigenvalues"""
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = n - i - 1
        A[i + 1, i] = i + 1
    return A


def write_suitesparse_mat(A, path: str) -> None:
    """
    Write a given sparse matrix as a SuiteSparse MATLAB .mat file.
    """
    problem = np.empty((1, 1), dtype=[("A", object)])
    problem["A"][0, 0] = sp.csc_matrix(A)
    scipy.io.savemat(path, {"Problem": problem})


def main():
    parser = argparse.ArgumentParser(
        description="Compare partial_schur against ARPACK on a SuiteSparse matrix."
    )
    parser.add_argument("mat_file", help="Path to the .mat file (SuiteSparse format)")
    parser.add_argument(
        "--nev",
        type=int,
        default=6,
        help="Number of eigenvalues to compute (default: 6)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-8, help="Convergence tolerance (default: 1e-8)"
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Maximum Krylov space dimension / ARPACK ncv (default: max(2*nev+1, 20))",
    )
    parser.add_argument(
        "--max-it",
        type=int,
        default=10_000,
        help="Maximum number of restarts (default: 10000)",
    )
    parser.add_argument(
        "--p", type=int, default=None, help="P value in Krylov-Schur (default: k+7)"
    )
    parser.add_argument(
        "--which",
        choices=list(WHICH_TO_SORT),
        default="LM",
        help="Which eigenvalues to target: LM (largest magnitude) "
        "or LR (largest real part). Default: LM",
    )
    args = parser.parse_args()

    nev = args.nev
    tol = args.tol
    max_it = args.max_it
    which = args.which
    sort_function = WHICH_TO_SORT[which]
    p = args.p

    if p is None:
        p = nev + 7

    # Load and prepare the matrix
    A_raw = load_suitesparse_mat(args.mat_file)
    n = A_raw.shape[0]
    nnz = A_raw.nnz
    # Convert to complex128 for a fair comparison: partial_schur only supports
    # complex types, so we cast both solvers to the same dtype.
    A = A_raw.astype(np.complex128)

    max_dim = args.max_dim if args.max_dim is not None else min(max(2 * nev + 1, 20), n)

    print(f"Matrix: {args.mat_file}")
    print(f"  shape={n}x{n}, nnz={nnz}, dtype={A.dtype}")
    print(
        f"  nev={nev}, tol={tol}, max_dim={max_dim}, "
        f"max_restarts={max_it}, which={which}"
    )

    # ------------------------------------------------------------------
    # ARPACK
    # ------------------------------------------------------------------
    print(f"\n--- Running ARPACK ---")
    arpack_counter = MatvecCounter(A)
    t0 = time.perf_counter()
    arpack_vals, arpack_vecs = eigs(
        arpack_counter,
        k=nev,
        which=which,
        ncv=max_dim,
        tol=tol,
        maxiter=max_it,
    )
    arpack_elapsed = time.perf_counter() - t0

    # Sort by descending real part for consistent display
    idx = np.argsort(-arpack_vals.real)
    arpack_vals = arpack_vals[idx]
    arpack_vecs = arpack_vecs[:, idx]

    matvecs = arpack_counter.matvecs
    n_iters = (matvecs - max_dim) // (max_dim - nev)
    print(f"  matvecs={arpack_counter.matvecs}, elapsed={arpack_elapsed:.2f}s for {n_iters} iterations")

    # ------------------------------------------------------------------
    # partial_schur
    # ------------------------------------------------------------------
    print(f"\n--- Running partial_schur (p = {p}) ---")
    ps_counter = MatvecCounter(A)
    t0 = time.perf_counter()
    pQ, pT, history = partial_schur(
        ps_counter,
        nev,
        max_dim=max_dim,
        stopping_criterion=tol,
        max_restarts=max_it,
        sort_function=sort_function,
        p=p,
    )
    ps_elapsed = time.perf_counter() - t0

    # Extract eigenpairs from the partial Schur form
    ps_eig_vals, S = np.linalg.eig(pT)
    ps_eig_vecs = pQ @ S

    idx = np.argsort(-ps_eig_vals.real)
    ps_eig_vals = ps_eig_vals[idx]
    ps_eig_vecs = ps_eig_vecs[:, idx]

    ps_matvecs = int(np.max(history.matvecs))
    n_iters = np.max(history.restarts)
    print(f"  matvecs={ps_matvecs}, elapsed={ps_elapsed:.2f}s")
    print(f"  matvecs={ps_matvecs}, elapsed={ps_elapsed:.2f}s for {n_iters} iterations")

    # ------------------------------------------------------------------
    # True residuals
    # ------------------------------------------------------------------
    print_residuals("ARPACK", A, arpack_vals, arpack_vecs)
    print_residuals("partial_schur", A, ps_eig_vals, ps_eig_vecs)

    # ------------------------------------------------------------------
    # Matvec comparison
    # ------------------------------------------------------------------
    arpack_mv = arpack_counter.matvecs
    pct = (ps_matvecs - arpack_mv) / arpack_mv * 100
    direction = "more" if pct >= 0 else "fewer"

    print(f"\n--- Matvec comparison ---")
    print(f"  ARPACK:        {arpack_mv} matvecs  ({arpack_elapsed:.2f}s)")
    print(f"  partial_schur: {ps_matvecs} matvecs  ({ps_elapsed:.2f}s)")
    print(f"  partial_schur uses {abs(pct):.1f}% {direction} matvecs than ARPACK")
    print(history)


if __name__ == "__main__":
    main()
