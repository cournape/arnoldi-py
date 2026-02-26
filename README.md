# Arnoldi

This is an attempt to write an eigensolver for sparse matrices that only relies
on NumPy and BLAS/LAPACK, without depending on ARPACK. Ultimately, the hope is
to be a viable replacement for scipy.sparse.linalg.eigs, and remove the fortran
dependency.

## How to install

The project is using uv. Assuming you have uv installed, simply do

``` shell
uv sync
```

## Example of usage

``` python
import numpy as np

from scipy.sparse.linalg import eigs

from arnoldi.krylov_schur import partial_schur
from arnoldi.matrices import mark
from arnoldi.utils import arg_largest_real


# Markov walk matrix
A = mark(50)

TOL = 1e-8
MAX_RESTARTS = 1_000
MAX_DIMS = 20
K = 5
WHICH = "LR"
SORT_FUNCTION = arg_largest_real

Q, T, history = partial_schur(
    A,
    K,
    max_dim=MAX_DIMS,
    stopping_criterion=TOL,
    sort_function=SORT_FUNCTION,
    max_restarts=MAX_RESTARTS,
)
# Convert Schur transform into eigenpairs
vals, S = np.linalg.eig(T)
vecs = Q @ S

print("========== true residuals, our implementation =============")
for k in range(K):
    val, vec = vals[k], vecs[:, k]
    residual = np.linalg.norm(A @ vec - val * vec)
    norm_residual = residual / np.abs(val)
    print(
        f"Residual {k}, value is {np.real(val):.7}, res is {residual:.3e}, norm res is {norm_residual:.3e}"
    )

# Compare to ARPACK
vals, vecs = eigs(A, K, ncv=MAX_DIMS, maxiter=MAX_RESTARTS, which=WHICH,
                  tol=TOL)

print("========== true residuals, ARPACK (scipy) implementation =============")
for k in range(K):
    val, vec = vals[k], vecs[:, k]
    residual = np.linalg.norm(A @ vec - val * vec)
    norm_residual = residual / np.abs(val)
    print(
        f"Residual {k}, value is {np.real(val):.7}, res is {residual:.3e}, norm res is {norm_residual:.3e}"
    )
```

It is not recommended for real use yet. I expect the API to change, especially
in terms of convergence tracking and history.

For more complex examples, look at the scripts [SLEPC
script](scripts/compare-against-slepc.py) and [ARPACK
script](scripts/compare-against-arpack.py).  It can compare ARPACK and SLEPc (to
be installed separately) against this implementation for matrices in the sparse
suite.

## Why ?

ARPACK-NG is a fortran library for sparse eigen solvers. It has the following issues:

- the fortran code is not thread-safe. In particular, it is not re-entrant
- it does not incorporate some of the more recent improvements discovered for
  large problems, e.g. A Krylov-Schur Algorithm for Large Eigenproblems, G. W.
  Stewart, SIAM J. Matrix Anal. Appl., Vol. 23, No. 3, pp. 601–614

## AI policy

AI tools such as Claude are extensively used for code review, literature review
and understanding alternative, license-compatible implementations.

However, as the code is a candidate to be incorporated into SciPy codebase, the
code itself is not generated using AI, and all the code has been manually
written from the literature. Exceptions are trivial scripts in the scripts/ directory. Files AI-generated
will contain a header mentioning this.

## TODO

For a first 1.0 release:

- [ ] Fundamental support for arbitrary matrices, largest eigen values only
  - [x] basic arnoldi decomp w/ tests
  - [x] add a key set of test matrices, using sparse matrix suite + synthetic
      (Markov, Laplace, etc.)
  - [x] convergence tracking on Ritz values
  - [x] explicit restart support with deflation
  - [x] Krylov-schur method
  - [ ] More robust convergence criteria (relative/absolute/A norm)
  - [x] customizable orthonormalization
- [ ] Compare performance w/ ARPACK in terms of #matvecs and runtime
  - [x] compare MGS vs double GS w/ DGKS vs others in terms of precision
  - [ ] implement locking and dynamic p
  - [ ] handle happy breakdown in Krylov-Schur
- [ ] add support for calculation in real space for real matrices
- [ ] LinearOperator support

Post 1.0:

- [ ] optimize for the case of Hermitian/symmetric matrices (Lanczos)
- [ ] add support for shift-invert (arbitrary eigen values)
- [ ] add support for general inverses problems
- [ ] single precision support
- [ ] optimization:
  - [x] optimize orthonormalization
  - [ ] ensure memory space is mostly V and nothing else as function of input
  size
  - [ ] block Krylov-Schur ?

## Existing alternative implementations

- matlab:
  - [KrylovSchur](https://github.com/dingxiong/KrylovSchur). Warning: no license.
  - [Various implementations of Lanczos, including selective
  orthogonalization](https://sites.cs.ucsb.edu/~gilbert/cs240a/matlab/eigenvals/).
  Warning: no license.
- julia
  - [Complete toolkit in pure julia](https://github.com/Jutho/KrylovKit.jl)
    - includes linsolve, expm in addition to eigen value solvers
  - [Faithful reimplementation of ARPACK in pure julia](https://github.com/dgleich/GenericArpack.jl)
  - [The Arnoldi method with Krylov-Schur restart, natively in pure Julia](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl/)
    - According to
    [https://discourse.julialang.org/t/ann-arnoldimethod-jl-v0-4/110604](https://discourse.julialang.org/t/ann-arnoldimethod-jl-v0-4/110604),
    it is more stable than ARPACK.
    - implementation ~ 2k julia (not including tests)
    - license compatible with SciPy (MIT-like)

## References

- Mathematical references:
  - [Lecture Notes on Solving Large Scale Eigenvalue
  Problems](https://people.inf.ethz.ch/arbenz/ewp/Lnotes) by Prof. Dr. Peter
  Arbenz from the Computer Science Department at ETH. See in particular the
  [latest version of the
  notes](https://people.inf.ethz.ch/arbenz/ewp/Lnotes/lsevp.pdf)
  - [Templates for the Solution of Algebraic Eigenvalue Problems: a Practical
  Guide](https://www.netlib.org/utk/people/JackDongarra/etemplates/book.html).
  Overview on numerical methods for eigen values, including dense and sparse
    - [A shifted block Lanczos algorithm for solving sparse symmetric generalized eigenproblems.](https://www.nas.nasa.gov/assets/nas/pdf/techreports/1991/rnr-91-012.pdf)
    - [Applied Numerical Linear Algebra](http://www.stat.uchicago.edu/~lekheng/courses/302/demmel/)
    - [Implicit application of polynomial filters in a k-step Arnoldi method](https://ntrs.nasa.gov/api/citations/19930004220/downloads/19930004220.pdf)
      - [An Implicitly Restarted Lanczos Method for Large Symmetric
      Eigenvalue Problems](http://etna.mcs.kent.edu/vol.2.1994/pp1-21.dir/pp1-21.ps):
      a specialization of the implicit ARNOLDI to the hermitian operator case.
    - [Thick-Restart Lanczos Method For Large Symmetric Eigen Values
    Problems](https://sdm.lbl.gov/~kewu/ps/trlan-siam.pdf): a special case of
    Krylov-Schur for Hermitian operators
    - [NUMERICAL METHODS FOR LARGE EIGENVALUE
    PROBLEMS](https://www-users.cse.umn.edu/~saad/eig_book_2ndEd.pdf): 2nd
    edition, only covers up to early 1990s techniques (explicit/implicit
    restarted Arnoldi). It explains clearly deflation, locking and gives some
    numerical examples that can be used as reference.
