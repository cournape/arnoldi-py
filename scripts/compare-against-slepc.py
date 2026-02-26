# AI note: this script was mostly generated using AI (claude code)
"""
Compute the K eigenvalues with largest real part of a SuiteSparse .mat matrix.

Usage
-----
  mpirun -n 4 python slepc_top_eigs.py --mat af23560.mat --k 10
  python slepc_top_eigs.py --mat af23560.mat --k 10 --ncv 60 --tol 1e-10

Dependencies
------------
  petsc4py, slepc4py, scipy, numpy

Notes
-----
  SuiteSparse .mat files store the matrix under key 'Problem' as a struct,
  with the sparse matrix itself at Problem['A'][0,0].  This loader handles
  that layout as well as the simpler case where the matrix is stored directly
  under a plain key.
"""

import argparse
import os.path
import sys
import time

import numpy as np
import scipy.sparse as sp


HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from utils import (
    WHICH_TO_SORT_SLEPC, ConvergenceTracker, EigensolverParameters, Statistics,
    load_suitesparse_mat, slepc_eig, print_residuals,
)


def parse_args() -> argparse.Namespace:
    # Strip PETSc/SLEPc flags (start with -) before argparse sees them
    clean = [a for a in sys.argv[1:] if not a.startswith("-eps")
                                     and not a.startswith("-st")
                                     and not a.startswith("-ksp")
                                     and not a.startswith("-pc")]
    parser = argparse.ArgumentParser(
        description="Compute top-K eigenvalues (largest real part) via SLEPc"
    )
    parser.add_argument("mat_file", help="Path to .mat file")
    parser.add_argument("--nev",    type=int, default=6,
                        help="Number of eigenvalues to compute (default: 6)")
    parser.add_argument("--max-dim",type=int, default=None,
                        help="Krylov subspace dimension (default: max(3k, k+32))")
    parser.add_argument("--tol",    type=float, default=1e-8,
                        help="Convergence tolerance (default: 1e-8)")
    parser.add_argument("--max-it", type=int, default=10_000,
                        help="Maximum iterations (default: 10000)")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disable per-iteration convergence output")
    parser.add_argument("--which", choices=list(WHICH_TO_SORT_SLEPC), default="LM",
                        help="Which eigenvalues to target: LM (largest magnitude) "
                             "or LR (largest real part). Default: LM")
    parser.add_argument(
        "--p", type=int, default=None, help="P value in Krylov-Schur (unused)"
    )
    return parser.parse_args(clean)


def main():
    args = parse_args()

    # ── Load matrix ─────────────────────────────────────────────
    A = load_suitesparse_mat(args.mat_file)
    A = A.astype(np.complex128)
    m, n = A.shape
    nnz  = A.nnz

    if m != n:
        raise ValueError("Matrix must be square for a standard eigenproblem.")

    parameters = EigensolverParameters.from_cli_args(args, n)
    print(f"Matrix: {args.mat_file}")
    print(f"  shape={n}x{n}, nnz={nnz}, dtype={A.dtype}")
    print(parameters)

    tracker = None if args.no_monitor else ConvergenceTracker()
    vals, vecs, stats = slepc_eig(A, parameters, tracker)

    print_residuals("SLEPC", A, vals, vecs)
    print("--- Perf comparison --")
    print(f"  SLEPC: {stats.matvecs} matvecs in {stats.restarts} iterations  ({stats.elapsed:.2f}s)")


if __name__ == "__main__":
    main()
