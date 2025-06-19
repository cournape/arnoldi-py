# Arnoldi

This is an attempt to write an eigensolver for sparse matrices that only relies
on NumPy and BLAS/LAPACK, without depending on ARPACK. Ultimately, the hope is
to be a viable replacement for scipy.sparse.eigen, and remove the fortran
dependency.

## Why ?

ARPACK-NG is a fortran library for sparse eigen solvers. It has the following issues:

- the fortran code is not thread-safe. In particular, it is not re-entrant
- it does not incorporate some of the more recent improvements discovered for
  large problems, e.g. A Krylov-Schur Algorithm for Large Eigenproblems, G. W.
  Stewart, SIAM J. M ATRIX A NAL. A PPL ., Vol. 23, No. 3, pp. 601–614

## TODO

- [ ] Basic support for symmetric/hermitian matrices, largest eigen values only
    - [ ] add a basic set of test matrices, using sparse matrix suite + synthetic
    - [ ] add basic unrestarted Lanczos algorithm
    - [ ] add convergence tracking on Ritz values
    - [ ] add block + full reortho
    - [ ] try implicit / explicit restart (notebooks)
    - [ ] add locking / deflate
    - [ ] partial reorthogonalization
    - [ ] try krylov-schur method
- [ ] add support for shift-invert (arbitrary eigen values)
- [ ] extend to Hermitian/symmetric matrices (Arnoldi)

## Existing alternative implementations

- matlab:
    - [KrylovSchur](https://github.com/dingxiong/KrylovSchur). Warning: no license.
    - [Various implementations of Lanczos, including selective orthogonalization](https://sites.cs.ucsb.edu/~gilbert/cs240a/matlab/eigenvals/). Warning: no license.
- julia
    - [Complete toolking in pure julia](https://github.com/Jutho/KrylovKit.jl)
        - includes linsolve, expm in addition to eigen value solvers
    - [Faithful reimplementation of ARPACK in pure julia](https://github.com/dgleich/GenericArpack.jl)
    - [The Arnold method with Krulov-Schur restart, natively in pure Julia](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl/)
        - According to [https://discourse.julialang.org/t/ann-arnoldimethod-jl-v0-4/110604](https://discourse.julialang.org/t/ann-arnoldimethod-jl-v0-4/110604), it is more stable than ARPACK.
        - implementation ~ 2k julia (not including tests)
        - license compatible with SciPy (MIT-like)

## References

- Explanations of ARPACK implementation: https://dgleich.micro.blog/2021/04/21/a-small-project.html (arpack faithful reimplementation in julia)
- Mathematical references:
	- [Lecture Notes on Solving Large Scale Eigenvalue Problems](https://people.inf.ethz.ch/arbenz/ewp/Lnotes) by Prof. Dr. Peter Arbenz from the Computer Science Department at ETH. See in particular the [latest version of the notes](https://people.inf.ethz.ch/arbenz/ewp/Lnotes/lsevp.pdf)
	- [Templates for the Solution of Algebraic Eigenvalue Problems: a Practical Guide](https://www.netlib.org/utk/people/JackDongarra/etemplates/book.html). Overview on numerical methods for eigen values, including dense and sparse
    - [A shifted block Lanczos algorithm for solving sparse symmetric generalized eigenproblems.](https://www.nas.nasa.gov/assets/nas/pdf/techreports/1991/rnr-91-012.pdf)
    - [Applied Numerical Linear Algebra](http://www.stat.uchicago.edu/~lekheng/courses/302/demmel/)
    - [Implicit application of polynomial filters in a k-step Arnoldi method](https://ntrs.nasa.gov/api/citations/19930004220/downloads/19930004220.pdf)
        - [An Implicitly Restarted Lanczos Method for Large Symmetric Eigenvalue Problems](http://etna.mcs.kent.edu/vol.2.1994/pp1-21.dir/pp1-21.ps):
          a specialization of the implicit ARNOLDI to the hermitian operator case.
    - [Thick-Restart Lanczos Method For Large Symmetric Eigen Values Problems](https://sdm.lbl.gov/~kewu/ps/trlan-siam.pdf): a special case of Krylov-Schur for Hermitian o
perators
    - [NUMERICAL METHODS FOR LARGE EIGENVALUE PROBLEMS](https://www-users.cse.umn.edu/~saad/eig_book_2ndEd.pdf): 2nd
      edition, only covers up to early 90ies techniques (explicit/implicit
      restarted Arnoldi). It explains clearly deflation, locking and gives some
      numerical examples that can be used as reference.
