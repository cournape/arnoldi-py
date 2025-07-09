import numpy as np
import scipy.sparse as sp
import scipy.linalg as slin

from scipy.io import loadmat
from arnoldi import ArnoldiDecomposition
from arnoldi.decomposition import RitzDecomposition

from numpy.linalg import norm

from tests.test_matrices import mark


N = 100
NEV = 5
K = 30 # 2 * NEV + 10
m = sp.random(N, N, density=5 / N, dtype=np.complex128)
m += sp.diags_array(np.ones(N))

N = 10
NEV = 3
K = 6
m = sp.diags_array(np.arange(1, N+1, 1)).astype(np.complex128)


def load_mhd1280b():
    d = loadmat("matrices/mhd1280b_SVD.mat")
    svds = d["S"][0][0][0]

    d = loadmat("matrices/mhd1280b.mat")
    m = d["Problem"][0][0][2]

    return m, np.squeeze(svds)


def load_suitesparse_from_name(name):
    d = loadmat(f"{name}_SVD.mat")
    svds = d["S"][0][0][0][:, 0]
    n = svds.shape[0]

    d = loadmat(f"{name}.mat")
    data = d["Problem"][0][0]
    for v in data:
        if v.shape == (n, n):
            break
    else:
        raise ValueError(f"Could not find expected sparse matrix for {name}")

    return v, svds


def arnoldi_simple(a, k=None):
    n = a.shape[0]
    nev = 1 # Naive arnoldi w/o restart only really works for 1 eigenvalue
    k = k or min(max(2 * nev + 1), n)

    arnoldi = ArnoldiDecomposition(n, k)
    arnoldi.initialize()

    n_iter = arnoldi.iterate(a)

    ritz = RitzDecomposition.from_arnoldi(arnoldi, nev, n_iter)
    #residuals = ritz.compute_residuals(a)
    #approx_residuals = arnoldi._approximate_residuals(nev, n_iter)

    return ritz.values, ritz.vectors


# TODO: does the following API make sense ?
#
# - converged defined as residual <= max(eps(type) * ||H||_fro, tol *
#   abs(ritz_value)) for a given ritz value / residual pair (see
#   ArnoldiMethod.jl julia package, try to find reference
# 
# This enables user to track convergence in both converged / non-converged
# cases, and control convergence/break
#
#    history = {} # iter -> residuals
#    solver = ....
#    init_vector = None
#    for i in range(max_iters):
#        solver.iterate(a, init_vector)
#        # Detect convergence
#        if has_converged:
#            break
#    else:
#        raise ValueError("Has not converged")
#    
#    ritz_values, ritz_vecs = solver.extract_ritz()
#    solver.matmuls

# TODO
# - add convergence calculation ala ArnoldiMethod.jl
# - implement basic ERAM w/ interface class, compare w/ AM for main eigenvalue
# - implement ERAM w/ locking/deflation, for > 1 eigenvalues
# - implement block ERAM ?
# - implement basic Krylov-Schur

import line_profiler

@line_profiler.profile
def arnoldi_with_naive_restart(a, max_iters, k=None):
    n = a.shape[0]
    nev = 1 # Arnoldi w/ naive restart only really works for one eigen value
    k = k or min(max(2 * nev + 1, 20), n)
    arnoldi = ArnoldiDecomposition(n, k)

    v = np.random.randn(n)
    v /= slin.norm(v)

    residuals_history = {}

    for i in range(max_iters + 1):
        if i > 0:
            arnoldi.q.fill(0)
            arnoldi.h.fill(0)
        arnoldi.initialize(v)

        n_iter = arnoldi.iterate(a)

        ritz = RitzDecomposition.from_arnoldi(arnoldi, nev, k=n_iter)
        if np.abs(ritz.approximate_residuals[0]) < arnoldi.atol:
            break

        v = ritz.vectors[:, 0]
        v /= slin.norm(v)

        residuals_history[i] = ritz.approximate_residuals

    return ritz.values, ritz.vectors, residuals_history, i


def min_distance(a, b):
    # Compute distance between a and b, ignoring differences due to sign
    # difference of conjugate
    m = np.inf
    for left in (a, np.conj(a)):
        for right in (b, np.conj(b)):
            m = min(m, np.abs(left - right))
            m = min(m, np.abs(left + right))

    return m


def fortran_d_format(x):
    if x == 0:
        return "0.000D+00"
    else:
        exponent = int(np.floor(np.log10(abs(x))))
        mantissa = x / (10 ** exponent)
        # Normalize mantissa to [0.1, 1.0)
        mantissa /= 10
        exponent += 1
        return f"{mantissa:.3f}D{exponent:+03d}"


def format_fortran_array(arr):
    return np.array2string(
        arr,
        formatter={'float_kind': fortran_d_format}
    )


F = fortran_d_format


if True:
    # Reproduce table 6.1 and 6.2 of NUMERICAL METHODS FOR LARGE
    # EIGENVALUE PROBLEMS, 2nd edition
    TOL = np.sqrt(np.finfo(np.float64).eps)

    A = mark(10)
    print(A.shape)
    #A, _ = load_mhd1280b()
    N = A.shape[0]

    r_values = np.sort(sp.linalg.eigs(A, NEV)[0])[::-1]
    print(r_values)

    print("=============== Simple Arnoldi w/o restarts =============")
    for k in [5, 10, 15, 20, 25, 30]:
        ritz_vals, ritz_vecs = arnoldi_simple(A, k)
        print(f"residual @ k = {k}       {F(min_distance(ritz_vals[0], r_values[0]))}")

    print("=============== Simple Arnoldi w/ restarts =============")
    k = 5
    for max_restarts in [10 // k, 20 // k, 30 // k, 40 // k, 50 // k]:
        ritz_vals, ritz_vecs, history, _ = arnoldi_with_naive_restart(A,
                                                                      max_restarts,
                                                                      k)
        print(f"residual @ k = {k} / {max_restarts * k} {F(min_distance(ritz_vals[0], r_values[0]))}")

    k = 20
    max_iters = 5000
    ritz_vals, ritz_vecs, history, n_iter = arnoldi_with_naive_restart(A,
                                                                       max_iters,
                                                                       k)
    print(f"Took {n_iter} iterations and ~{n_iter * k} mat vecs to converge")
    print(f"residual        {min_distance(ritz_vals[0], r_values[0]):.5e}")


if False:
    A, _ = load_suitesparse_from_name("matrices/af23560")
    N = A.shape[0]
    #A += (-40.+260j) * sp.eye(N)

    NEV = 20
    MAX_ITERS = 5
    K = min(max(NEV * 2 + 1, 20), N)

    print(f"Problem size is {N}, finding {NEV} eigen values, building Krylov of size {K}")

    ritz_vals, ritz_vecs, history, n_iter = arnoldi_with_naive_restart(A, k=50,
                                                              max_iters=800)
    print(n_iter)
    residuals = slin.norm(A @ ritz_vecs - ritz_vals * ritz_vecs, axis=0)
    print(f"Took {n_iter} iterations and ~{n_iter * K} mat vecs to converge")
    print(f"residuals       {residuals[0]:.5e}")
    print(ritz_vals[0])
    print(f"Using ARPACK (scipy)")
    r_vals, r_vecs = sp.linalg.eigs(A, k=NEV, ncv=50, tol=1e-4)
    print(f"diff            {min_distance(ritz_vals[0], r_vals[0]):.5e}")
    print(r_vals[0])
    residuals = slin.norm(A @ r_vecs[:, :NEV] - r_vals * r_vecs[:, :NEV], axis=0)
    for i in range(NEV):
        print(r_vals[i])
        print(f"{residuals[i]:.5e}")
