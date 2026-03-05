# Krylov-Schur and Cross-Product SVD in SLEPc: Implementation Analysis

**SLEPc Code Analysis Report**
Repository: [gitlab.com/slepc/slepc](https://gitlab.com/slepc/slepc)
Commit: [`2435073`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/)

---

## 1. Krylov-Schur Eigensolver

### 1.1 Background and Notation

Given a large sparse matrix $A \in \mathbb{C}^{n \times n}$, we seek $k$ eigenpairs
$(\lambda_i, x_i)$ satisfying

$$A x_i = \lambda_i x_i, \qquad i = 1, \ldots, k, \quad k \ll n.$$

The Krylov-Schur method, introduced by Stewart [2001], maintains a *Krylov-Schur
decomposition* of order $m$:

$$A V_m = V_m H_m + \beta_{m} v_{m+1} e_m^*, \tag{KSD}$$

where $V_m \in \mathbb{C}^{n \times m}$ has orthonormal columns, $H_m \in
\mathbb{C}^{m \times m}$ is upper Hessenberg (or tridiagonal in the Hermitian case),
$\beta_m = \|f_m\|_2$ is the residual norm, $v_{m+1} = f_m / \beta_m$ is the
residual vector (next Krylov basis direction), and $e_m$ is the last standard basis
vector.

The key insight of Krylov-Schur is that a *partial Schur decomposition* of $H_m$,

$$H_m Q_m = Q_m S_m,$$

with $S_m$ upper quasi-triangular (real Schur form) or upper triangular (complex),
can be *truncated cheaply* to restart: retaining only the leading $p < m$ columns of
$Q_m$ gives a new valid Krylov-Schur decomposition of order $p$ with no additional
matrix-vector products.

### 1.2 Setup Phase

The solver entry point is
[`EPSSetUp_KrylovSchur`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L93),
which selects among several variants depending on the problem type:

| Condition | Variant | Solve function |
|---|---|---|
| Hermitian, standard extraction | `EPS_KS_SYMM` | `EPSSolve_KrylovSchur_Default` |
| Non-Hermitian or harmonic | `EPS_KS_DEFAULT` | `EPSSolve_KrylovSchur_Default` |
| Two-sided | `EPS_KS_TWOSIDED` | `EPSSolve_KrylovSchur_TwoSided` |
| Spectrum slicing (`EPS_ALL`) | `EPS_KS_SLICE` | `EPSSolve_KrylovSchur_Slice` |
| Generalized indefinite | `EPS_KS_INDEF` | `EPSSolve_KrylovSchur_Indefinite` |

The DS (Direct Solver) object type is set accordingly:
[`DSNHEP`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L160)
for the non-Hermitian case (Hessenberg eigenproblem) or
[`DSHEP`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L168)
(compact tridiagonal) for the Hermitian case.

The three dimension parameters are:

- `ncv`: maximum size of the working subspace (number of columns of $V_m$)
- `nev`: number of wanted eigenpairs
- `mpd`: maximum projected dimension, $\text{mpd} = \text{ncv} - \text{nev}$ at most

They are initialized at
[line 115](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L115)
via `EPSSetDimensions_Default`.

Key context fields (defined in
[`krylovschur.h`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.h#L78)):

```c
typedef struct {
  PetscReal keep;   /* restart parameter: fraction of basis to keep, default 0.5 */
  PetscBool lock;   /* locking (default: true) vs non-locking variant            */
  ...
} EPS_KRYLOVSCHUR;
```

### 1.3 Main Solve Loop

The core algorithm is implemented in
[`EPSSolve_KrylovSchur_Default`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L231).
We describe each step with the corresponding source location.

---

#### Step 1: Initialization

[Line 249](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L249)

```c
PetscCall(EPSGetStartVector(eps, 0, NULL));
l = 0;
```

A normalized starting vector $v_1$ is placed in column 0 of the BV (Basis Vectors)
object `eps->V`. The variable `l` tracks the number of "kept" vectors from the
previous restart (initially zero).

---

#### Step 2: Expand the Arnoldi/Lanczos Factorization

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
[`STGetOperator`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L259).

---

#### Step 3: Solve the Projected Eigenproblem

[Lines 277–285](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L277)

```c
PetscCall(DSSolve(eps->ds, eps->eigr, eps->eigi));
PetscCall(DSSort(eps->ds, eps->eigr, eps->eigi, eps->rr, eps->ri, pj));
PetscCall(DSUpdateExtraRow(eps->ds));
PetscCall(DSSynchronize(eps->ds, eps->eigr, eps->eigi));
```

`DSSolve` computes the Schur decomposition of $H_m$ (or $T_m$):

$$H_m = Q_m S_m Q_m^*,$$

where $S_m$ is upper quasi-triangular (real Schur form). The eigenvalues (Ritz values)
$\tilde{\lambda}_i$ are the diagonal entries of $S_m$. `DSSort` reorders the decomposition
so that the wanted eigenvalues come first (e.g., largest magnitude). `DSUpdateExtraRow`
updates the last row of the factored form (necessary for residual estimates after
the basis transformation).

---

#### Step 4: Convergence Check

[Line 288](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L288)

```c
PetscCall(EPSKrylovConvergence(eps, PETSC_FALSE, eps->nconv, nv-eps->nconv,
                               beta, 0.0, gamma, &k));
```

For each Ritz pair $(\tilde{\lambda}_i, \tilde{x}_i = V_m y_i)$ where $y_i$ is the
$i$-th column of $Q_m$, the residual norm estimate is computed cheaply as:

$$\|A\tilde{x}_i - \tilde{\lambda}_i \tilde{x}_i\|_2 \approx \beta_m \, |e_m^* y_i|,$$

which requires only the last component of each Schur vector $y_i$ (implemented in
[`epskrylov.c:208`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/epskrylov.c#L208)).
A pair is declared converged when $\|r_i\| / |\tilde{\lambda}_i| < \varepsilon_{\text{tol}}$
(relative criterion) or similar user-selected criterion.

The number of converged pairs $k$ is returned.

---

#### Step 5: Compute the Restart Length `l`

[Lines 294–298](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L294)

```c
l = PetscMax(1, (PetscInt)((nv - k) * ctx->keep));
if (!hermitian) PetscCall(DSGetTruncateSize(eps->ds, k, nv, &l));
```

The restart retains $p = k + l$ columns, where:

- $k$ = converged pairs (locked, if locking variant is used)
- $l$ = additional kept directions, $l \approx \lfloor (m_v - k) \cdot \theta \rfloor$

with restart parameter $\theta \in [0.1, 0.9]$ (default $\theta = 0.5$, set at
[line 126](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L126)).

For the non-Hermitian case, `DSGetTruncateSize` adjusts `l` to ensure the truncation
point falls between a $2 \times 2$ real Schur block boundary (preserving conjugate
pairs of complex eigenvalues).

---

#### Step 6: Truncate the Schur Decomposition (Restart)

[Line 326](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L326)

```c
PetscCall(DSTruncate(eps->ds, k + l, PETSC_FALSE));
```

This is the defining operation of Krylov-Schur: the Schur decomposition is truncated
to its leading $p = k + l$ columns. The result is a new valid Krylov-Schur
decomposition

$$A V_p = V_p H_p + \beta_m v_{m+1} e_p^* Q_{p:m}^*,$$

where $H_p$ is still upper quasi-triangular and the new residual vector is a linear
combination of the old one. No additional matrix-vector products are needed for
the truncation itself.

---

#### Step 7: Update the Basis Vectors

[Lines 330–335](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L330)

```c
PetscCall(DSGetMat(eps->ds, DS_MAT_Q, &U));
PetscCall(BVMultInPlace(eps->V, U, eps->nconv, k + l));
PetscCall(DSRestoreMat(eps->ds, DS_MAT_Q, &U));
...
PetscCall(BVCopyColumn(eps->V, nv, k + l));  /* copy restart vector */
```

The physical basis is updated as $V_p \leftarrow V_m Q_{:,1:p}$, a dense
$n \times m$ times $m \times p$ multiplication. The restart vector $v_{p+1}$ is
set to $v_{m+1}$ (the last Arnoldi/Lanczos vector), which is the proper continuation
direction in the new compressed subspace.

---

#### Step 8: Locking

[Lines 299, 345](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/eps/impls/krylov/krylovschur/krylovschur.c#L299)

In the **locking** variant (default, `ctx->lock = PETSC_TRUE`), converged eigenpairs
are deflated: columns $0, \ldots, k-1$ of $V_m$ are fixed and orthogonality against
them is enforced in all subsequent Arnoldi steps.

```
eps->nconv = k;
```

In the **non-locking** variant, $k$ is reset to zero so that all directions remain
in the working subspace, at the cost of computing all residuals every iteration.

---

### 1.4 Algorithm Summary (Pseudocode)

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

### 1.5 Key Parameters

| Parameter | API | Default | Effect |
|---|---|---|---|
| Subspace size `ncv` | `EPSSetDimensions` | `max(2·nev, 10)` | Larger → fewer restarts, more memory |
| Restart fraction `keep` | `EPSKrylovSchurSetRestart` | `0.5` | Fraction of $m-k$ directions kept |
| Locking | `EPSKrylovSchurSetLocking` | `PETSC_TRUE` | Lock converged pairs |
| Extraction | `EPSSetExtraction` | `EPS_RITZ` | Ritz or harmonic Ritz values |

---

## 2. Block Krylov-Schur

The standard Krylov-Schur method is a *single-vector* method: the Krylov subspace is
generated by repeated application of $A$ to a single starting vector. The *block*
variant generalizes this by working with a block of $b$ starting vectors simultaneously.

### 2.1 Block Arnoldi Decomposition

Let $\mathcal{V}_m = [V_1, V_2, \ldots, V_m]$ where each $V_j \in \mathbb{C}^{n \times b}$.
The block Arnoldi decomposition satisfies

$$A \mathcal{V}_m = \mathcal{V}_m \mathcal{H}_m + V_{m+1} B_m E_m^*,$$

where:

- $\mathcal{H}_m \in \mathbb{C}^{mb \times mb}$ is block upper Hessenberg
- $V_{m+1} \in \mathbb{C}^{n \times b}$ is the next block of orthonormal vectors
- $B_m \in \mathbb{R}^{b \times b}$ is upper triangular (from QR of the residual block)
- $E_m = I_{mb} \otimes e_b$ (last $b$ columns of $I_{mb}$)

Each block step applies $A$ to all $b$ columns:
$\hat{V}_{j+1} = A V_j$, then orthogonalizes against all previous blocks
$\mathcal{V}_j$ and computes a QR factorization $\hat{V}_{j+1} \leftarrow V_{j+1} B_j$.

### 2.2 Block Krylov-Schur Restart

The Schur decomposition of $\mathcal{H}_m$ gives

$$\mathcal{H}_m = \mathcal{Q}_m \mathcal{S}_m \mathcal{Q}_m^*,$$

and truncation to the leading $p$ columns (with $p$ a multiple of $b$) yields:

$$A \mathcal{V}_p = \mathcal{V}_p \mathcal{H}_p + V_{m+1} B_m E_p^* \mathcal{Q}_{p:mb}^*.$$

The new residual matrix is $F_p = V_{m+1} B_m \mathcal{Q}_{mb, 1:p}^*$, with
$\|F_p\| = \|B_m \mathcal{Q}_{mb, 1:p}^*\|$. The Schur truncation remains valid
because the block structure is preserved.

### 2.3 Advantages of the Block Variant

1. **Multiple starting vectors**: targets clusters of eigenvalues or eigenspaces
   with multiplicity $\geq b$ more robustly (single-vector methods may miss
   degenerate eigenvalues).

2. **Better BLAS-3 performance**: the dominant cost shifts from BLAS-2 (matrix-vector
   products for orthogonalization) to BLAS-3 (matrix-matrix products), which have
   much higher arithmetic intensity on modern hardware.

3. **Improved convergence for clustered eigenvalues**: the block method
   implicitly works with the entire invariant subspace rather than one eigenvector
   at a time, so clustered or repeated eigenvalues converge simultaneously.

4. **Parallelism**: each block step performs $b$ independent matrix-vector products,
   which can be pipelined or batched in a distributed setting.

### 2.4 Implementation Considerations in SLEPc

To extend SLEPc's `EPSSolve_KrylovSchur_Default` to support block iterations:

1. **`BVMatArnoldi` → block version**: replace the single-vector Arnoldi call with
   a block variant that applies `STApply` to all $b$ columns of the current block
   and orthogonalizes against the full basis using `BVOrthogonalize`. SLEPc's BV
   infrastructure (particularly `BVMult`, `BVDot`, `BVOrthogonalize`) already
   operates on block slices and would accommodate this with minimal changes.

2. **DS object**: the `DSNHEP` dense solver would receive a block upper Hessenberg
   matrix. The Schur decomposition computed by `DSSolve` works on the full matrix
   regardless of block structure, so no change is required there.

3. **Convergence criterion**: the per-vector residual $\beta_m |e_m^T y_i|$ generalizes
   to $\|B_m e_{m,b}^T Y_i\|$ where $Y_i$ is the block of Schur vectors and $e_{m,b}$
   selects the last $b$ rows.

4. **`DSTruncate`**: truncation remains valid block-by-block; the truncation index
   should be aligned to a block boundary $p \equiv 0 \pmod{b}$.

5. **Restart vector**: instead of a single $v_{p+1}$, the restart produces a block
   $V_{p/b+1}$ that must be orthonormalized (QR factorization of $F_p$).

---

## 3. Default Sparse SVD Solver: SVDCROSS

### 3.1 Background

Given a sparse matrix $A \in \mathbb{C}^{m \times n}$, the singular value
decomposition is

$$A = U \Sigma V^*, \qquad \Sigma = \mathrm{diag}(\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0),$$

where $U \in \mathbb{C}^{m \times m}$ and $V \in \mathbb{C}^{n \times n}$ are unitary.
We seek the $k$ largest (or smallest) singular triplets $(\sigma_i, u_i, v_i)$.

### 3.2 Cross-Product Reduction

The default SVD solver in SLEPc is `SVDCROSS`, selected at
[`svdsetup.c:244`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/interface/svdsetup.c#L244):

```c
if (!((PetscObject)svd)->type_name) PetscCall(SVDSetType(svd, SVDCROSS));
```

The fundamental mathematical reduction is:

$$A^* A \, v_i = \sigma_i^2 \, v_i, \tag{CEP}$$

i.e., the singular values of $A$ are square roots of eigenvalues of the Hermitian
positive semidefinite matrix $C = A^*A$, and the right singular vectors $v_i$ are
the corresponding eigenvectors. Left singular vectors are recovered as
$u_i = A v_i / \sigma_i$.

### 3.3 Setup Phase

The setup is in
[`SVDSetUp_Cross`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L162).

**Step 1 — Build the cross product operator.**
[Lines 176–196](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L176)

```c
PetscCall(SVDCrossGetProductMat(svd, svd->A, svd->AT, &cross->C));
```

By default (`explicitmatrix = PETSC_FALSE`), a *shell matrix* $C$ is created:

```c
MatMult_Cross(B, x, y):
    w ← A * x
    y ← A^T * w       /* y = A^T A x */
```

[Lines 31–41](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L31).
This avoids forming $A^*A$ explicitly (which would be denser and potentially
much larger than $A$ itself), at the cost of two matrix-vector products per
eigenvalue iteration step instead of one.

The explicit matrix option (`-svd_cross_explicitmatrix`) computes $C = A^*A$
numerically using PETSc's `MatProductCreate` infrastructure
([lines 114–132](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L114)).

**Step 2 — Configure the subsidiary EPS.**
[Lines 199–236](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L199)

```c
PetscCall(EPSSetOperators(cross->eps, cross->C, NULL));
PetscCall(EPSSetProblemType(cross->eps, EPS_HEP));
PetscCall(EPSSetWhichEigenpairs(cross->eps, EPS_LARGEST_MAGNITUDE));
```

The internal EPS (eigensolver) is configured to solve the Hermitian eigenproblem
$C v = \lambda v$ for the largest eigenvalues (or smallest, depending on `svd->which`).
The default EPS type is `EPSKRYLOVSCHUR` (the solver described in Section 1), so
`SVDCROSS` is ultimately a two-level method: Krylov-Schur on $A^*A$.

The tolerance is set tighter by a factor of 10 to account for the squaring of
singular values in the convergence criterion:
[line 217](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L217).

### 3.4 Solve Phase

[`SVDSolve_Cross`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L256)
simply delegates to the subsidiary eigensolver:

```c
PetscCall(EPSSolve(cross->eps));
PetscCall(EPSGetConverged(cross->eps, &svd->nconv));
for (i = 0; i < svd->nconv; i++) {
  PetscCall(EPSGetEigenvalue(cross->eps, i, &lambda, NULL));
  svd->sigma[i] = PetscSqrtReal(PetscRealPart(lambda));   /* σᵢ = √λᵢ */
}
```

[Lines 264–280](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L264).

Mathematically, the relationship is:

$$\sigma_i = \sqrt{\lambda_i(A^*A)}, \qquad i = 1, \ldots, k.$$

A guard against floating-point negative eigenvalues (due to roundoff in $A^*A$)
is present at line 273.

### 3.5 Vector Recovery Phase

[`SVDComputeVectors_Cross`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L284)

Right singular vectors $v_i$ are the eigenvectors from EPS, stored directly:

```c
PetscCall(EPSGetEigenvector(cross->eps, i, v, NULL));
```

Left singular vectors are recovered via
[`SVDComputeVectors_Left`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L352):

$$u_i = \frac{A v_i}{\|A v_i\|_2} = \frac{A v_i}{\sigma_i}.$$

This requires one additional matrix-vector product per converged singular triplet.

### 3.6 Algorithm Summary

```
Input:  A ∈ ℂᵐˣⁿ, target nsv singular triplets
Output: Singular triplets (σᵢ, uᵢ, vᵢ)

1.  Build C = A*A (shell matrix or explicit)
2.  Configure EPS (default: EPSKRYLOVSCHUR) for C vᵢ = λᵢ vᵢ
3.  EPSSolve(eps):   [runs Krylov-Schur on C, Section 1]
4.  for i = 1..nconv:
5.    σᵢ ← √λᵢ      (eigenvalue → singular value)
6.    vᵢ ← eigenvector of C  (right singular vector)
7.    uᵢ ← A vᵢ / σᵢ        (left singular vector, one matvec)
```

### 3.7 Analysis: Strengths and Limitations

**Strengths:**

- Simple implementation — reuses the full EPS machinery, including shift-and-invert,
  spectrum slicing, and parallel scalability.
- Shell matrix avoids forming $A^*A$ explicitly: memory footprint remains $O(\text{nnz}(A))$.
- Works for generalized SVD ($A$, $B$) and hyperbolic SVD ($\Omega$-weighted) with
  minimal changes (different operator construction).

**Limitations:**

- The condition number of $A^*A$ is $\kappa(A)^2$, so convergence of the
  eigensolver is sensitive to ill-conditioning: eigenvalues near zero (small
  singular values) are computed with relative accuracy $O(\varepsilon_{\text{mach}} \cdot \kappa(A)^2)$.
- Two matrix-vector products per iteration step (once through $A$, once through $A^*$)
  compared to one in a direct Lanczos bidiagonalization approach (e.g., `SVDTRLANCZOS`).
- The SVD technical report in SLEPc notes: *"In the case of standard SVD, the
  computation done by this solver will be almost identical to the one with* `SVDTRLANCZOS`"
  — both ultimately perform Lanczos on $A^*A$ through the shell matrix
  (see [`cross.c:622`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/src/svd/impls/cross/cross.c#L622)).

---

## 4. References

1. G.W. Stewart, *A Krylov-Schur Algorithm for Large Eigenproblems*, SIAM J. Matrix
   Anal. Appl., 23(3):601–614, 2001.

2. V. Hernandez, J.E. Roman, A. Tomas, V. Vidal, *Krylov-Schur Methods in SLEPc*,
   SLEPc Technical Report STR-7, Universitat Politecnica de Valencia, 2007.

3. V. Hernandez, J.E. Roman, A. Tomas, *Practical Implementation of Harmonic
   Krylov-Schur*, SLEPc Technical Report STR-9, 2009.

4. V. Hernandez, J.E. Roman, A. Tomas, V. Vidal, *A Survey of Software for Sparse
   Eigenvalue Problems*, SLEPc Technical Report STR-6, 2009.

5. Y. Saad, *Numerical Methods for Large Eigenvalue Problems*, SIAM, 2011.

6. G. Golub, C. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins, 2013.

---

*All source links point to commit
[`2435073`](https://gitlab.com/slepc/slepc/-/blob/2435073368006cab65837fb206144f409508c908/)
of the SLEPc GitLab repository.*
