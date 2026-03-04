# How I used AI for write this module ?

Goal is to write a competitive OSS sparse eigensolver under BSD-compatible
license.

Structure
- Background:
  - What is eigenvalue: compute invariants (generalized by Schur transform)
  - Closely related to SVD
  - Applications:
    - physical problems (find example)
    - graph representation ->
    - low rank approximation -> PageRank, recommendation (collaborative
    filtering, Netflix prize), large linear regression, etc.
      - Note: only in the ||^2 sense. But if you care about sparse solutions,
      ||^1 or even ||^0 -> compressed sensing, Donoho/Candes/Tao
- What is the problem ?
  - Typical algorithm for dense matrices: iterated QR algorithm on Hessenberg form
    - require multiplying the original matrix: expensive and not realistic for sparse matrices (kills sparsity)
    - often for large systems you only care about a few eigen pairs, not all
    - actually, often a partial Schur transform is enough
  - scipy has an existing implementation for sparse, but in arcane Fortran
  (literatly translated to C). Difficult to understand, debug or improve.
  Cannot benefit from optimizations for heterogenous computing (e.g. GPU/CPU)
    - cannot "just write using AI" a nice version because of copyright
- Basics of Krylov-Schur:
  - power method
  - reuse intermediate vector -> Krylov basis
  - Arnoldi: you compute an orthonormal basis of Krylov basis (more stable), and that also gives an Hessenberg approximation (low rank) of A
- I leveraged initially ChatGPT, and then Claude code to write a competitive implementation
  - literature review: ask to find the references, and then adding the papers in the claude project, ask questions for specifics I did not understand
  - solve nagging issues to install reference implementations: example of SLEPc
  installation
  - ask to summarize existing, license compatible codebases
    - example of SLEPc: prompt to understand the exact locking logic
  - ask claude to generate benchmarking scripts:
    - benchmarking / plotting
  - Review my code based on test cases and output
    - review code that had bugs. Example of session w/ wrong output + referring
    to my implementation
      - the complex conjugate issue: was stuck for a long time on convergence
      issues that only happen in specific cases (specific matrices, and only after
      a specific number of steps) -> claude code (opus 4.5 at that time ?) could
      run the code on different cases and suggest several solutions, including the
      correct one (forgot to conjugate one vector in some calculation)
      - finding issue w/ dynamic p
  - ask to review invariants to implement tests
