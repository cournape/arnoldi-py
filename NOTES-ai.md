# Using AI to build a competitive sparse eigen solver

## What is this presentation about?

1. **How to use agentic coding to assist scientific code** while still writing the code
2. **Show concrete examples of where agentic coding works well and less well**:
   including literature review linked to code, debugging convergence issues,
   help with benchmarking/testing, and comparison against other implementations.
3. **Learn a bit about numerical computation**: specifically how to compute
   eigenpairs and SVD of large, sparse systems (millions of rows, thousands of
   parameters), and also a bit of history.

As a preview "aha" moment of agentic coding: for a 3rd party OSS library
implementing in C the algorithm I was interested in, I could generate in
minutes a 15-page document describing the relevant implementation step by
step, with annotated source code. All I had to do was clone the repo locally
and describe in a few sentences what details I wanted, including potential
extensions not yet implemented. The generated document
contains a section-by-section description of the code, interlacing math
explanations with it, e.g.

---

[Lines 256–268](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L256)

The target subspace size for this restart is

$$m_v = \min(\text{nconv} + \text{mpd},\ \text{ncv}).$$

Starting from column `nconv + l` (after converged vectors and kept vectors),
the factorization is extended to $m_v$ columns:

**Non-Hermitian case** — Arnoldi process via `BVMatArnoldi`:

$$A V_m = V_m H_m + \beta_m v_{m+1} e_m^*, \quad H_m \in \mathbb{C}^{m \times m} \text{ upper Hessenberg},$$

**Hermitian case** — three-term Lanczos via `BVMatLanczos`:

$$A V_m = V_m T_m + \beta_m v_{m+1} e_m^*, \quad T_m \in \mathbb{R}^{m \times m} \text{ tridiagonal}.$$

The spectral transformation (ST) operator $\text{Op} = K(A)$ (e.g., shift-and-invert
$(A - \sigma I)^{-1}$) is applied transparently through
[`STGetOperator`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L259)

---

The lines link to the code for reference. I can also follow up on parts I
don't understand within the same Claude Code session, whether on the math or
the code, and ask how my own implementation differs from it.

I know there is a lot of hype around AI and "AI fatigue", but this is such a
powerful tool for understanding and learning. I know my PhD would have been
completely different with a tool like that.

## A bit of context

Finding eigenvalues/eigenvector and SVD of matrices is a key algorithm.  It is
useful for many tasks:

1. Used to find low rank approximation of large matrices:
   1. PCA in data analysis
   2. non-negative matrix factorization, e.g. for collaborative filtering
     (recommendation)
   3. spectral clustering (used in scikit learn)
2. Network analysis
   1. Graph Laplacian: the second smallest eigenvalue of the graph Laplacian is
   0 iff the graph is disconnected
   2. Random walk: PageRank (Google's original algorithm) finds the stationary
   distribution of a random walk via the dominant eigenvector of the
   transition matrix
3. Many more applications in physics, etc.

In the simple case, we say $\lambda, v$ is an eigenpair for the matrix $A$ if:

  $$ A v = \lambda v $$

$\lambda$ is the eigenvalue, and $v$ an eigenvector. Conceptually, it means $v$
is an invariant for $A$, i.e. $v$ stays in the same direction after applying
the linear operator of $A$. It is closely related to Singular Value
Decomposition (SVD).

Sparse eigensolver are algorithms which can find only a few eigen pairs of a
potentially very large, sparse (non zero entries << zeros, generally 1 % or
less). E.g. original netflix prize matrix was ~20k x 500k, oroginal "Google
Matrix" of the 1998 page rank paper ~ 24 millions x 24 millions.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

n = 5000
density = 0.001

A = sp.random(n, n, density=density)

# Find the two eigenpairs with the largest (L) magnitude (M) eigenvalues
eigenvalues, eigenvectors = spla.eigs(A, k=2, which="LM")
```

I had had this project for ~15 years to rewrite a better version, but never
found the time. In 2024, my friend Stefan van der Walt reminded me of those
discussions, and I used some downtime at Scipy 2024 conference to try to get
ChatGPT to give me the steps needed for a state-of-the-art implementation. One
twist: for copyright reasons, I wanted to write the code myself, understand the
algorithm completely, and not ask the AI to write the code for me.

So my goals were:

1. **Implement a SOTA sparse eigensolver**: at least as fast as scipy's one,
   but in Python and easy to extend
2. **Do it in a couple of days' worth of work**: not part of my current job
   obviously, and I am not a grad student anymore
3. **Do not generate code directly**: the hope is to incorporate it in scipy
   proper. Cannot do this if I do not own the copyright.

Key point: agentic coding is useful even under this constraint. **Thanks to
ChatGPT and then claude code, I could complete this in a couple of days, from
literature research to competitive implementation** even as I still wrote all
the non-boilerplate code!

As a bonus, I am now familiar with the basics of sparse eigensolver methodology.

### How did I use AI for this section?

First, basic "chat" use. First attempt in summer 2024, I started using ChatGPT
for literature review: finding references and websites, and having it help me
understand algorithms based on descriptions. Example prompt: "What is
deflation, and how does it relate to locking for explicit restarts?" on top of
a screenshot of a given algorithm. I would ask it to explain a given section if
I did not understand, and to expand on it. **This helped me figure out whether this was
feasible at all given my constraints**.

Claude code also helped me find two bugs, one easy and one subtle:

1. Easy: 360a0e39dad3de019343656950e000d1e0b9c3b2
2. More subtle:  985d9911bb4dfef255d0f860f576d4639a4d2635, vdot vs dot, conjugate

How did I find them? First one by asking claude code to review the code
directly.

Second one, more interesting: I wrote a script that reproduced the issue,
printed its output, and asked claude code to find the bug by trying different
matrices and parameters, with examples that worked and one that did not. It ran
in an agentic mode and finally found that based on the matrix type (real vs.
complex), there was a missing conjugate when handling convergence. That was a
big "aha" moment for me. Copying/pasting the code into ChatGPT to find the bug did
not work, and the point is easily glossed over when you look at an algorithm
description. I made a few attempts in 2025 to fix this bug, and would have
likely given up if it were not for claude code.

**AGENTIC AI LESSON**: use CC to help you understand details of algorithms you
don't understand. Use it within the context of your projects, and leverage CC
web search to find the right info.

**AGENTIC AI LESSON**: leverage agentic "prompt CC -> run verbose script
showing issue -> review by CC" to find bugs.

## Writing a competitive implementation and more advanced uses of claude code

By Jan 2026, having fixed the basic implementation, I doubled down on CC and
started using it more systematically.

### Generating benchmarking scripts

This one is straightforward. After having a basic Krylov Schur, I wanted to
compare it against the `scipy` one, both in terms of performance and accuracy. I asked it to create a script that could

1. Read an existing sparse matrix from the sparse suite (a set of sparse
   matrices coming from various fields)
2. Run both scipy and my implementation over a set of parameters ($m$, number
   of eigenvalues to find, etc.) defined in a config file. Compare the output
   in terms of precision and benchmark their timing
3. Write a script to plot the results for easier comparison, see below

![](https://private-user-images.githubusercontent.com/25111/556924723-d5c082a8-0c0a-446a-8232-2fba2467c3ad.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzI3MjExMDIsIm5iZiI6MTc3MjcyMDgwMiwicGF0aCI6Ii8yNTExMS81NTY5MjQ3MjMtZDVjMDgyYTgtMGMwYS00NDZhLTgyMzItMmZiYTI0NjdjM2FkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzA1VDE0MjY0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ5M2ZmYmMyOTM3Yjc0Y2NjZmY5NThmZTg3NjZhMTdjMTZjNDhjNDM3NTAxYWMwZDJjYWZiOThlNzdlYWVmYTYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.VtKR6yaKAnGKehJ4eYZw5kZG4ncVria_nGG90fmgEN0)

Now, [those](https://github.com/cournape/arnoldi-py/blob/main/scripts/plot-stress-test.py) [scripts](https://github.com/cournape/arnoldi-py/blob/main/scripts/stress-test.py) are fairly trivial, but it would have easily taken me 1h+ to write them.

### Automatically figuring out how to install complex libraries

I mentioned earlier SLEPc, a modern C implementation of various sparse solvers
(eigenvalues, but also linear systems, SVD, etc.). As is typical of complex
software, installing those libraries from sources with python bindings is a
major PITA.

Instead, I asked claude code to install SLEPc in a virtualenv and write down
the instructions once it had figured it out. For security reasons (downloading
and installing from the internet) I did this in a temporary VM. After ~20
mins, it produced [the instructions](https://github.com/cournape/arnoldi-py/blob/main/scripts/INSTALL_SLEPC.md)

Now, as a former build engineer, I know how to do this, but it would again
easily have taken me 1+ hour to figure it out, given the compilation time and
the weird incompatibilities. Also note that it figured out automatically how to
use the right environment variables (easy enough), and the right version of
Cython, since the latest Cython is not compatible with the source code (it
figured this out from the build output).

### Help writing tests

Generally, I find CC not so good at writing unit tests when directed from an existing
implementation. It will often generate trivial code, mock unnecessarily, etc.
Instead, I asked CC to confirm the important invariants to check: V orthonormal,
equality of $A V - H V$ and approximate residuals, etc. Then I wrote [the tests
based on those
invariants](https://github.com/cournape/arnoldi-py/blob/main/tests/test_decomposition.py#L36),
though CC could of course have generated those itself.

**AGENTIC AI LESSON**: be careful when writing tests from the code being tested. It
will result in bad tests. Think "write specifications of the tests", then
review, then ask CC to write tests from the specifications to check.

**Note**: this is a key principle of successful agent/LLM usage, agentic coding or
otherwise. Those tools are really good when checking a solution can be done more
easily or faster than finding one, because then agents can loop in the
background and figure it out by themselves. If, however, it is very difficult to
check the LLM output, and even more so at scale, then LLMs will most likely not
work very well. There are similarities to LLM/agent evals.

### Finding another non-trivial convergence bug

Another mind-blowing moment was CC finding a non-trivial bug.
The Krylov-Schur method has a step where it truncates a transform of the Hessenberg
matrix. For simplicity, I started with a fixed truncation factor $p$, which
worked well enough. However, dynamic $p$ (as more pairs converge) is more
efficient. Unfortunately, following the same logic as SLEPc caused a complete
breakdown of convergence.

This time, I simply asked CC the following:

```
When I run with a dynamic p, my code break down. See e.g. this output example

# Here I copy-pasted the verbose output of a fail run
...

1. Can you confirm the code for dynamic p logic is correct ? Compare with SLEPc
   logic and find any discrepency
2. If the code for p logic is correct, can you find why convergence is failing
   in the above case ?
```

It found that while my logic for $p$ was correct, the convergence failure
was due to a corner case where $p$ may not always monotonically increase. In this
case, as I was reusing some buffer of a smaller size, some values in the buffer
were "dirty".

This is again fairly mind-blowing because previous attempts at finding the bug
from source alone did not work. It is only once it had the verbose output that
it saw that $p$ may decrease between iterations, and that my code was reusing
some dirty values. It also suggested to log some additional invariants to find
issues.

### Getting contextual, more detailed information about implementation

Existing documentation is often not enough to understand some subtle details of
an implementation. SLEPc technical reports had some missing information I could
get from the code, but this takes time.

After cloning the SLEPc source code, and using the following prompt:

```
  Based on this SLEPC git clone, create a report in markdown w/ the following:

  1. analyze the code for Krylov Schur, explain the algorithm with math notation, linking every step to the correponding section. Use github link to the latest commit for
  each link
  2. add a section in how to expand the algorithm to do block krylov schur
  3. do like 1 for the default sparse SVD solver of this library

  Write the report in the same style as SLEPc tech reports
```

I got this type of content:

---

```
Input:  A ∈ ℂⁿˣⁿ, target nev eigenpairs, subspace size ncv, mpd, keep θ
Output: Converged Ritz pairs (λ̃ᵢ, ṽᵢ)

1.  v₁ ← random normalized vector; l ← 0; nconv ← 0
2.  while not converged:
3.    m ← min(nconv + mpd, ncv)
4.    Extend V from nconv+l to m columns via Arnoldi/Lanczos:
          A Vₘ = Vₘ Hₘ + βₘ vₘ₊₁ eₘᵀ
5.    Compute Schur decomp:  Hₘ = Qₘ Sₘ Qₘ*
6.    Sort Schur form (wanted eigenvalues first)
7.    Update extra row of Schur form
8.    for i = nconv to m:
9.      rᵢ ← βₘ |eₘᵀ yᵢ|     (cheap residual estimate)
10.     if rᵢ / |λ̃ᵢ| < tol:  mark converged → k ← k+1
11.   l ← max(1, floor((m - k) · θ))  (kept directions)
12.   Truncate Schur form to p = k+l columns:  DSTruncate(k+l)
13.   Update physical basis:  Vₚ ← Vₘ · Qₘ[:,1:p]
14.   Set restart vector:  vₚ₊₁ ← vₘ₊₁
15.   nconv ← k  (lock converged pairs)
```

---

and also

---

```
Step 4: Convergence Check

Line 288

PetscCall(EPSKrylovConvergence(eps, PETSC_FALSE, eps->nconv, nv-eps->nconv,
                               beta, 0.0, gamma, &k));

For each Ritz pair
where is the -th column of , the residual norm estimate is computed cheaply as:

which requires only the last component of each Schur vector (implemented in epskrylov.c:208). A pair is declared converged when
(relative criterion) or similar user-selected criterion.

The number of converged pairs is returned.
```
---

This completely blew my mind. In particular, the main algorithm description has
more details than the technical report (exact convergence criteria, exact logic
for m and l), and those details come from the code. The section-by-section
explanation of the C code, with details extracted from the underlying
implementation in other files, really aids understanding. It acts as a kind of
reverse literate programming.

**AGENTIC AI LESSON**: use them to summarize or deep-dive into codebases that
can act as a reference or that you need to understand for your work. It is
specifically good at linking multiple references together, and is now very good
at generating math through LaTeX.

**Note**: in this specific case, you definitely want to use OPUS, not sonnet. I
was trying to reproduce the output without success until I realized I was using
a weak model.

## Conlusion

1. Start using a coding agent now. You don't need configuration to start using
   it. Seeing and using is believing
2. Start use it for scripting, one-off actions, things you know how to do by
   hand so that you can verify the output, and are drudgery.
3. Leverage the agentic loop: give it a task that can be done by looping over a
   well setup (e.g. a script). Then it can do the work for you in the background
4. It is not useful only for writing code, but also to interact with code:
   understand a codebase, ask specific implementation questions, review your
   code

## Notes and references

**Note:** what is LAPACK? BLAS / LAPACK are a set of routines (functions),
originally written in Fortran in the 1970s, for linear algebra. E.g. the BLAS
function dgemv is a function to compute y = A x + b given A, x and b. Those
libraries can be written in very optimized code, and every CPU architecture
used to provide an optimized implementation. Those could be 10x faster,
sometimes even more, compared to a typical implementation, thanks to very
low-level optimization (SIMD, etc.). For example, Apple provides the Accelerate
framework that implements those functions for the M* architecture, Intel the
MKL, CUDA for NVIDIA (GPU). OpenBLAS, an OSS implementation, tends to be
competitive on CPU.

**Note**: scipy incorporates various libraries written in Fortran, sometimes in the
1970s! In that time, there were no screens but teletypes, and you would submit
your code in batches through punchcards. Those punchcards had a specific format
of 80 columns, and to this day many editors default to 80 columns. Modern
Fortran is actually a decent language for numerical computing, but many old
libraries are full of goto and other constructs used when for loops were not
common. [Example](https://github.com/scipy/scipy/blob/6e246d0b54dd55dc69232a0caae6772228a7ac25/scipy/integrate/odepack/lsoda.f) if you want to be scared.

## Some background on sparse eigen decomposition

### eigen decomposition and its applications

Why is this useful?

1. Used to find low rank approximation of large matrices:
   1. PCA in data analysis
   2. non-negative matrix factorization, e.g. for collaborative filtering
     (recommendation)
   3. spectral clustering (used in scikit learn)
2. Network analysis
   1. Graph Laplacian: the second smallest eigenvalue of the graph Laplacian is
   0 iff the graph is disconnected
   2. Random walk: PageRank (Google's original algorithm) finds the stationary
   distribution of a random walk via the dominant eigenvector of the
   transition matrix
3. Many more applications in physics, etc.

### Eigen solver, sparse matrices

An eigensolver is an algorithm that can numerically compute the eigenpairs of
a matrix. E.g. numpy.linalg.eig, which uses the underlying LAPACK library.
This algorithm is O(N^3), and works well if you want all the eigenpairs and
you work with dense matrices.

In some applications, you want either 1) to find only a couple of eigenpairs
(largest, smallest, or the ones closest to a given region of the complex plane)
or 2) you can't store the full matrix because it is too big, i.e. it is sparse
(number of != 0 entries is small, typically 1 % or less).

Many practical applications meet those two conditions. For example:

1. Collaborative filtering: you have an N x M matrix, for N users and M items,
   and each entry contains the user score. You want to predict the score for
   a new (user, movie) pair, which can be done through factorization /
   completion. The matrix is sparse (any user has only watched a couple of
   movies), and completion through low-rank approximation (e.g. top 100
   eigenpairs) is feasible. M is maybe 10k, and N is maybe 100e6. The full
   matrix would be ~7.5 TB, and it would take forever to run an O(n^3) algorithm.
2. Finding the largest eigenpair of the Google matrix (PageRank): the Google
   matrix is defined such that column j represents where a random user would go
   after visiting web page j, i.e. N x N where N is the number of pages on the
   web (oversimplified).

Most numerical packages (numpy/scipy, matlab, octave, mathematica) use ARPACK,
a Fortran library to find a few eigenpairs of a sparse matrix.

### Why write a new solver?

ARPACK, like many Fortran libraries, is written in arcane Fortran, which is
difficult to maintain. Also, today we want to leverage heterogeneous hardware
(e.g. GPU), and if it is written in Fortran, it becomes a black box that is
hard to run on new hardware.

### Basics of sparse eigensolver

In this section, we will review the basics of a set of algorithms called
Krylov-based methods. We will also explain how I used ChatGPT to do literature
review, learn details of algorithms, and use claude code to debug a convergence
issue.

#### Power method

The algorithms we discuss are so-called Krylov-based methods, which build on
the power method. The power method can help find the eigenpair for the largest eigenvalue
if $|\lambda| > 1$. The basic idea is simple:

  Algorithm:

  1. Choose a random starting vector $v_0 \in \mathbb{R}^n$, $|v_0| = 1$
  2. For $k = 1, 2, \ldots$ until convergence:

  $$z_k = A v_{k-1}$$
  $$v_k = \frac{z_k}{|z_k|}$$

This algorithm has a key advantage: it does not need to "know" $A$, but only
how to calculate the function $f(x) = A x$. As long as you can define this
function, the algorithm works. If the function is easy to distribute, then you
can find the top eigenvalue/eigenvector on a cluster of machines (initial
PageRank partitioned the underlying graph).

#### Krylov basis

The power method is not efficient because at every step, it "throws away" the
previous estimate $v_k$. It is more practical to consider the Krylov basis
$\mathcal{K}_m(A, v)$:

$$\mathcal{K}_m(A, v) = \mathrm{span}\left\{ v, Av, A^2 v, \ldots, A^{m-1} v \right\}$$

Unfortunately, there is a problem building a Krylov basis from a numerical
perspective. As the power of A increases, $A^p v_{p-1}$ and $v_{p-1}$ become increasingly likely to be (numerically)
collinear, which means you get "stuck".

#### Arnoldi decomposition

Many modern algorithms are based on Arnoldi decomposition. Arnoldi
decomposition iteratively computes an orthonormal basis of the Krylov space.

After $m$ steps, the Arnoldi decomposition reads:

 $$A V_m = V_m H_m + h_{m+1,m}\, v_{m+1} e_m^T$$

where:

  - $V_m = \lbrack v_1 \mid v_2 \mid \cdots \mid v_m \rbrack \in \mathbb{R}^{n \times m}$
  has orthonormal columns spanning $\mathcal{K}_m(A, v_1)$
  - $H_m \in \mathbb{R}^{m \times m}$ is upper Hessenberg
  - $h_{m+1,m}$ is the next off-diagonal entry
  - $e_m \in \mathbb{R}^m$ is the $m$-th canonical basis vector

Equivalently, multiplying on the right by $V_m^T$:

  $$H_m = V_m^T A V_m$$

Note that $H_m$ is much smaller than $A$, as long as $m \ll n$. Finding the
eigenpairs of $H_m$ as a dense matrix is doable. $m$ is generally a few times
the number of eigenpairs you are interested in. E.g. to find the top 50 eigenpairs of a one million by one million matrix,
using $m \approx 200$ is enough, so $H_m$ is $200 \times 200$, and even $V_m$
is still manageable on a decent machine. The eigenpairs of $H_m$ are called
Ritz pairs of $A$. There are theoretical justifications that Ritz values are
good approximations of $A$'s eigenvalues, and Ritz vectors projected back to
$A$'s space through $V_m$ are eigenvectors.

#### Beyond Arnoldi

Arnoldi has two limitations:

1. You need to increase $m$ if you want more precision
2. You can only find the first eigenpair

The first problem is solved through *restarts*: if convergence is not achieved
after a limit $m_1$, the algorithm uses the latest vector as the new vector
$v_0$ and restarts an Arnoldi decomposition "from scratch". The convergence rate
is somewhat slower per cycle, but the required size of $V$ (and thus the cost
of orthonormalization) remains bounded.

The second problem is harder to solve, and accounts for most of ARPACK's
complexity. After finding the first eigenpair, it creates a new Krylov space, but
*deflates* the already converged pair(s), i.e. it ensures the new Krylov space
does not contain the direction of the converged eigenvectors.

In the early 2000s, a new formulation called Krylov-Schur was discovered. It
is much simpler to implement. It is the default method used in SLEPc, and the
one I decided to implement.
