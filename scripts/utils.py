import dataclasses
import time

import numpy as np
import scipy.io
import scipy.sparse as sp

from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, eigs

from petsc4py import PETSc
from slepc4py import SLEPc

from arnoldi.krylov_schur import partial_schur
from arnoldi.utils import arg_largest_real, arg_largest_magnitude


WHICH_TO_SORT = {
    "LM": arg_largest_magnitude,
    "LR": arg_largest_real,
}

WHICH_TO_SORT_SLEPC = {
    "LM": SLEPc.EPS.Which.LARGEST_MAGNITUDE,
    "LR": SLEPc.EPS.Which.LARGEST_REAL,
}


@dataclasses.dataclass
class Statistics:
    elapsed: float = 0.0
    dtype: np.dtype = dataclasses.field(default_factory=lambda: np.dtype("complex128"))
    matvecs: int = 0
    restarts: int = 0


@dataclasses.dataclass
class EigensolverParameters:
    nev: int = 6
    ncv: int = 20
    tol: float = 1e-8
    max_restarts: int = 1_000
    p: int | None = None
    which: str = "LM"

    @classmethod
    def from_cli_args(cls, args, n):
        max_dim = args.max_dim if args.max_dim is not None else min(max(2 * args.nev + 1, 20), n)

        return cls(
            args.nev, max_dim, args.tol, args.max_it, args.p, args.which,
        )


class MatvecCounter(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.dtype = np.dtype(A.dtype)
        self.matvecs = 0

    def _matvec(self, x):
        self.matvecs += 1
        return self.A @ x

    def _rmatvec(self, x):
        self.matvecs += 1
        return self.A.conj().T @ x


class PETScMatvecCounter:
    def __init__(self, A):
        self.A = A
        self.matvecs = 0

    def mult(self, A_shell, x, y):
        self.matvecs += 1
        self.A.mult(x, y)

    def multTranspose(self, A_shell, x, y):
        self.matvecs += 1
        self.A.multTranspose(x, y)


def find_best_matching(a, b):
    """
    Reorder both arrays so that they match as closely as possible
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Compute pairwise distances
    # Cost matrix: |a[i] - b[j]| for all pairs
    cost_matrix = np.abs(a[:, np.newaxis] - b[np.newaxis, :])

    # Find optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Check that matched pairs are close
    return a[row_ind], b[col_ind]


def load_suitesparse_mat(path: str) -> sp.csr_matrix:
    """
    Load a SuiteSparse MATLAB .mat file.
    """
    data = scipy.io.loadmat(path, squeeze_me=False)

    # Try the SuiteSparse struct layout first
    prob = data.get("Problem")
    if prob:
        # prob is a (1,1) structured array; the matrix lives at field 'A'
        A = prob["A"][0, 0]
        if sp.issparse(A):
            return A.tocsr()

    raise ValueError(f"No sparse matrix found in {path!r}")


def compute_residuals(A, vals, vecs):
    for k, (val, vec) in enumerate(zip(vals, vecs.T)):
        res = np.linalg.norm(A @ vec - val * vec)
        norm_res = res / abs(val)


def print_residuals(label, A, vals, vecs):
    print(f"\n--- True residuals: {label} ---")
    for k, (val, vec) in enumerate(zip(vals, vecs.T)):
        res = np.linalg.norm(A @ vec - val * vec)
        norm_res = res / abs(val)
        print(
            f"  eigval[{k}] = {val.real:+.6g}{val.imag:+.6g}j"
            f"    |Av-\u03bbv|={res:.3e}    |Av-\u03bbv|/|\u03bb|={norm_res:.3e}"
        )


def arpack_eig(A, parameters: EigensolverParameters):
    A = MatvecCounter(A)
    t0 = time.perf_counter()

    vals, vecs = eigs(
        A,
        k=parameters.nev,
        which=parameters.which,
        ncv=parameters.ncv,
        tol=parameters.tol,
        maxiter=parameters.max_restarts,
    )
    elapsed = time.perf_counter() - t0

    # Sort by descending real part for consistent display
    idx = WHICH_TO_SORT[parameters.which](vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    matvecs = A.matvecs
    n_iters = (matvecs - parameters.ncv) // (parameters.ncv - parameters.nev)

    return vals, vecs, Statistics(elapsed, A.dtype, matvecs, n_iters)


def arnoldi_py_eig(A, parameters: EigensolverParameters):
    A = MatvecCounter(A)
    t0 = time.perf_counter()

    Q, T, history = partial_schur(
        A,
        parameters.nev,
        max_dim=parameters.ncv,
        stopping_criterion=parameters.tol,
        max_restarts=parameters.max_restarts,
        sort_function=WHICH_TO_SORT[parameters.which],
        p=parameters.p,
    )
    elapsed = time.perf_counter() - t0

    # Extract eigenpairs from the partial Schur form
    vals, S = np.linalg.eig(T)
    vecs = Q @ S

    idx = WHICH_TO_SORT[parameters.which](vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    matvecs = int(np.max(history.matvecs))
    n_iters = np.max(history.restarts)

    return vals, vecs, Statistics(elapsed, A.dtype, matvecs, n_iters)


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
    ctx = PETScMatvecCounter(A_petsc)
    A_shell = PETSc.Mat().createPython(
        A_petsc.getSizes(), context=ctx, comm=A_petsc.getComm()
    )
    A_shell.setUp()
    return A_shell, ctx


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
    mode = WHICH_TO_SORT_SLEPC[which]
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
    PETSc.Sys.Print(f"Solving for {k} eigenvalues with mode {mode} ({which})…")
    eps.solve()

    # ── Diagnostics ─────────────────────────────────────────────
    nconv = eps.getConverged()
    reason = eps.getConvergedReason()
    n_iter = eps.getIterationNumber()
    PETSc.Sys.Print(
        f"Converged reason : {reason}  ({n_iter} iterations)"
        f"\nConverged pairs  : {nconv} / {k} requested"
    )
    if nconv < k:
        PETSc.Sys.Print(
            f"WARNING: only {nconv} of {k} pairs converged.  "
            f"Try increasing --ncv or --max_it."
        )

    # ── Collect results ─────────────────────────────────────────
    results = []
    vr, vi = A_petsc.createVecs()
    for i in range(eps.getConverged()):
        k = eps.getEigenpair(i, vr, vi)
        error  = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
        results.append((k, vr.getArray().copy(), error))

    # Sort by descending real part (SLEPc usually returns them sorted, but
    # the standard does not guarantee it)
    idx = WHICH_TO_SORT[which]([_[0] for _ in results])
    results = [results[i] for i in idx]

    eps.destroy()
    return results


def slepc_eig(A, parameters: EigensolverParameters, tracker):
    comm = PETSc.COMM_WORLD
    A_petsc = scipy_csr_to_petsc(A, comm)
    A_shell, mv_ctx = wrap_with_matvec_counter(A_petsc)

    # ── Solve ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = solve_largest_real(
        A_shell,
        k= parameters.nev,
        max_dim=parameters.ncv,
        tol=parameters.tol,
        max_it=parameters.max_restarts,
        which=parameters.which,
        tracker=tracker,
    )
    elapsed = time.perf_counter() - t0

    matvecs = mv_ctx.matvecs
    restarts = tracker.history[-1]["iter"]

    stats = Statistics(elapsed, np.dtype(PETSc.ScalarType), matvecs, restarts)

    A_shell.destroy()
    A_petsc.destroy()

    vals = np.array([_[0] for _ in results])
    vecs = np.array([_[1] for _ in results]).T

    return vals, vecs, stats
