# arpack-py

This is an attempt to write an eigensolver for sparse matrices without
depending on ARPACK. Ultimately, the hope is to be a viable replacement for
scipy.sparse.eigen.arpack, and remove the fortran dependency.

## Why ?

ARPACK-NG is a fortran library for sparse eigen solvers. It has the following issues:

- the fortran code is not thread-safe. In particular, it is not re-entrant
- it does not incorporate some of the more recent improvements discovered for
  large problems, e.g. A Krylov-Schur Algorithm for Large Eigenproblems, G. W.
  Stewart, SIAM J. M ATRIX A NAL. A PPL ., Vol. 23, No. 3, pp. 601–614

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
