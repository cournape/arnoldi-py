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
import scipy.io
import scipy.sparse as sp

import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc


HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from utils import load_suitesparse_mat


WHICH_TO_SORT = {
    "LM": SLEPc.EPS.Which.LARGEST_MAGNITUDE,
    "LR": SLEPc.EPS.Which.LARGEST_REAL,
}

# ──────────────────────────────────────────────────────────────
# 1.  I/O helpers
# ──────────────────────────────────────────────────────────────

def scipy_csr_to_petsc(A_scipy: sp.csr_matrix, comm) -> PETSc.Mat:
    """
    Convert a scipy CSR matrix to a PETSc AIJ matrix.

    Works in both sequential (comm size 1) and parallel (MPI) runs.
    In parallel PETSc distributes rows automatically via createAIJ.
    """
    print(PETSc.ScalarType)
    A_scipy = A_scipy.tocsr()
    A_scipy.sort_indices()           # PETSc requires sorted column indices

    m, n = A_scipy.shape
    nnz_per_row = np.diff(A_scipy.indptr).astype(PETSc.IntType)

    A_petsc = PETSc.Mat().createAIJ(
        size=(m, n),
        csr=(A_scipy.indptr.astype(PETSc.IntType),
             A_scipy.indices.astype(PETSc.IntType),
             A_scipy.data.astype(PETSc.ScalarType)),
        comm=comm,
    )
    A_petsc.assemblyBegin()
    A_petsc.assemblyEnd()
    return A_petsc


# ──────────────────────────────────────────────────────────────
# 2.  Matvec counter (Python shell mat wrapping a PETSc AIJ mat)
# ──────────────────────────────────────────────────────────────

class MatvecCounter:
    """Context object used by PETSc's MATPYTHON shell."""

    def __init__(self, A_petsc: PETSc.Mat):
        self._A = A_petsc
        self.matvecs = 0

    def mult(self, A_shell, x, y):
        """y = A * x  — called for every forward matvec."""
        self.matvecs += 1
        self._A.mult(x, y)

    def multTranspose(self, A_shell, x, y):
        """y = A^T * x  — called when SLEPc needs the transpose."""
        self.matvecs += 1
        self._A.multTranspose(x, y)


def wrap_with_matvec_counter(A_petsc: PETSc.Mat) -> tuple[PETSc.Mat, MatvecCounter]:
    """
    Wrap *A_petsc* in a MATPYTHON shell that counts every matvec.

    Returns
    -------
    A_shell : PETSc.Mat
        Drop-in replacement for A_petsc to pass to SLEPc.
    ctx : MatvecCounter
        After solving, read ``ctx.matvecs`` for the total matvec count.
    """
    ctx = MatvecCounter(A_petsc)
    A_shell = PETSc.Mat().createPython(
        A_petsc.getSizes(), context=ctx, comm=A_petsc.getComm()
    )
    A_shell.setUp()
    return A_shell, ctx


# ──────────────────────────────────────────────────────────────
# 3.  Convergence monitor
# ──────────────────────────────────────────────────────────────

class ConvergenceTracker:
    """Stores per-iteration history for later analysis / plotting."""

    def __init__(self):
        self.history = []          # list of dicts, one per iteration

    def __call__(self, eps, its, nconv, evals, errs):
        self.history.append({
            "iter":   its,
            "nconv":  nconv,
            "evals":  list(evals),   # copy — SLEPc reuses the buffer
            "errors": list(errs),    # same
        })

        # Print a one-liner per iteration to stdout
        if its % 100 == 99:
            err_str = "  ".join(f"{e:.2e}" for e in errs[:4])  # show first 4
            PETSc.Sys.Print(
                f"  iter {its:4d} | nconv {nconv:3d} | errs [{err_str}]"
            )


# ──────────────────────────────────────────────────────────────
# 3.  Eigensolver
# ──────────────────────────────────────────────────────────────

def solve_largest_real(
    A_petsc:  PETSc.Mat,
    k:        int   = 6,
    max_dim:  int   = None,
    tol:      float = 1e-10,
    max_it:   int   = 500,
    which:    str   = 'LM',
    tracker:  ConvergenceTracker = None,
) -> list[tuple[complex, float]]:
    """
    Return the k eigenvalues with largest real part, together with their
    residual errors, sorted by descending real part.

    Parameters
    ----------
    A_petsc  : assembled PETSc matrix
    k        : number of eigenvalues to compute
    ncv      : Krylov subspace dimension (Arnoldi vectors kept).
               Rule of thumb: max(3*k, k+32).  Larger → more memory,
               faster convergence.
    tol      : convergence tolerance on the residual norm
    max_it   : maximum number of Arnoldi restarts
    which    : mode ('LR', 'LM', etc.)
    tracker  : optional ConvergenceTracker (monitor callback)

    Returns
    -------
    List of (eigenvalue, residual_error) tuples.
    """
    if max_dim is None:
        max_dim = max(3 * k, k + 32)
    if max_dim <= k:
        raise ValueError(f"ncv={max_dim} must be strictly greater than k={k}")

    eps = SLEPc.EPS().create(comm=A_petsc.getComm())

    # ── Problem definition ──────────────────────────────────────
    eps.setOperators(A_petsc)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)   # Non-Hermitian Eigenproblem

    # ── Which eigenvalues ───────────────────────────────────────
    mode = WHICH_TO_SORT[which]
    eps.setWhichEigenpairs(mode)

    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    #eps.setType(SLEPc.EPS.Type.ARNOLDI)
    # ── Subspace and convergence parameters ────────────────────
    eps.setDimensions(nev=k, ncv=max_dim)
    eps.setTolerances(tol=tol, max_it=max_it)

    # ── Convergence criterion: relative to eigenvalue magnitude ─
    eps.setConvergenceTest(SLEPc.EPS.Conv.REL)

    # ── Monitor ────────────────────────────────────────────────
    if tracker is not None:
        eps.setMonitor(tracker)

    # ── Allow command-line overrides (-eps_type, -eps_ncv, …) ──
    eps.setFromOptions()

    # ── Solve ───────────────────────────────────────────────────
    PETSc.Sys.Print(f"\nSolving for {k} eigenvalues with mode {mode} ({which})…")
    eps.solve()

    # ── Diagnostics ─────────────────────────────────────────────
    nconv = eps.getConverged()
    reason = eps.getConvergedReason()
    n_iter = eps.getIterationNumber()
    PETSc.Sys.Print(
        f"\nConverged reason : {reason}  ({n_iter} iterations)"
        f"\nConverged pairs  : {nconv} / {k} requested"
    )
    if nconv < k:
        PETSc.Sys.Print(
            f"WARNING: only {nconv} of {k} pairs converged.  "
            f"Try increasing --ncv or --max_it."
        )

    # ── Collect results ─────────────────────────────────────────
    results = []
    for i in range(nconv):
        eigval = eps.getEigenvalue(i)
        error  = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
        results.append((eigval, error))

    # Sort by descending real part (SLEPc usually returns them sorted, but
    # the standard does not guarantee it)
    results.sort(key=lambda x: x[0].real, reverse=True)

    eps.destroy()
    return results


# ──────────────────────────────────────────────────────────────
# 4.  Entry point
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # Strip PETSc/SLEPc flags (start with -) before argparse sees them
    clean = [a for a in sys.argv[1:] if not a.startswith("-eps")
                                     and not a.startswith("-st")
                                     and not a.startswith("-ksp")
                                     and not a.startswith("-pc")]
    parser = argparse.ArgumentParser(
        description="Compute top-K eigenvalues (largest real part) via SLEPc"
    )
    parser.add_argument("mat", help="Path to .mat file")
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
    parser.add_argument("--which", choices=list(WHICH_TO_SORT), default="LM",
                        help="Which eigenvalues to target: LM (largest magnitude) "
                             "or LR (largest real part). Default: LM")
    return parser.parse_args(clean)


def main():
    args = parse_args()
    comm = PETSc.COMM_WORLD

    # ── Load matrix ─────────────────────────────────────────────
    PETSc.Sys.Print(f"Loading {args.mat} …")
    A_scipy = load_suitesparse_mat(args.mat)
    A_scipy = A_scipy.astype(np.complex128)
    m, n = A_scipy.shape
    nnz  = A_scipy.nnz
    PETSc.Sys.Print(f"Matrix: {m} × {n},  {nnz} nonzeros")

    if m != n:
        raise ValueError("Matrix must be square for a standard eigenproblem.")

    # ── Convert to PETSc ────────────────────────────────────────
    A_petsc = scipy_csr_to_petsc(A_scipy, comm)
    A_shell, mv_ctx = wrap_with_matvec_counter(A_petsc)

    # ── Set up monitor ──────────────────────────────────────────
    tracker = None if args.no_monitor else ConvergenceTracker()

    # ── Solve ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = solve_largest_real(
        A_shell,
        k       = args.nev,
        max_dim = args.max_dim,
        tol     = args.tol,
        max_it  = args.max_it,
        which   = args.which,
        tracker = tracker,
    )
    elapsed = time.perf_counter() - t0

    # ── Print results ───────────────────────────────────────────
    PETSc.Sys.Print("\n" + "─" * 60)
    PETSc.Sys.Print(f"{'#':>3}  {'Real part':>18}  {'Imag part':>18}  {'Rel. error':>12}")
    PETSc.Sys.Print("─" * 60)
    for i, (eigval, err) in enumerate(results):
        PETSc.Sys.Print(
            f"{i:>3}  {eigval.real:>+18.10e}  {eigval.imag:>+18.10e}  {err:>12.4e}"
        )
    PETSc.Sys.Print("─" * 60)

    # ── Optionally summarise convergence history ─────────────────
    if tracker and tracker.history:
        total_iters = tracker.history[-1]["iter"]
        PETSc.Sys.Print(f"\nTotal monitored iterations : {total_iters}")
        # Convergence of first eigenpair over iterations
        PETSc.Sys.Print("\nFirst eigenpair error vs iteration:")
        for h in tracker.history[::max(1, len(tracker.history)//10)]:
            if h["errors"]:
                PETSc.Sys.Print(f"  iter {h['iter']:4d}  err {h['errors'][0]:.3e}")

    PETSc.Sys.Print(f"\nTotal matvecs against A : {mv_ctx.matvecs}")
    print(f"SLEPc: {mv_ctx.matvecs} matvecs, {h['iter']} iters in {elapsed:.2f}s ({PETSc.ScalarType})")

    A_shell.destroy()
    A_petsc.destroy()


if __name__ == "__main__":
    main()
