import numpy as np
import scipy.sparse as sp


def mark(m):
    """
    Generate a sparse Markov transition matrix for a random walk
    on a (lower) triangular grid with m rows.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse transition matrix of size n x n, where n = m*(m+1)/2.

    note
    ----
    See section 2.5.1 of Numerical methods for large eigenvalue problems, 2nd
    edition, and subsequent chapters for numerical examples of eigenvalues and
    convergence speed.
    """
    # FIXME: naive implementation, but we only use this for testing
    n = m * (m + 1) // 2
    cst = 0.5 / (m - 1)

    data = []
    row = []
    col = []

    ix = 0
    for i in range(m):
        jmax = m - i
        for j in range(jmax):
            ix += 1

            if j < jmax - 1:
                pd = cst * (i + j + 1)

                # North move
                jx = ix + 1
                data.append(pd)
                row.append(ix - 1)
                col.append(jx - 1)
                if i == 0:
                    data.append(pd)
                    row.append(ix - 1)
                    col.append(jx - 1)

                # East move
                jx = ix + jmax
                data.append(pd)
                row.append(ix - 1)
                col.append(jx - 1)
                if j == 0:
                    data.append(pd)
                    row.append(ix - 1)
                    col.append(jx - 1)

            # South/Up move
            pu = 0.5 - cst * (i + j - 1)
            if j > 0:
                jx = ix - 1
                data.append(pu)
                row.append(ix - 1)
                col.append(jx - 1)

            # West move
            if i > 0:
                jx = ix - jmax - 1
                data.append(pu)
                row.append(ix - 1)
                col.append(jx - 1)

    return sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()


def laplace_eigen(n):
    """ Returns the N eigen values of a Laplacian operator of dimension N.

    The Laplacian operator is the symmetric, tridiagonal matrix with -2 in the
    main diagonal, and 1 in the upper/lower diagonal.
    """
    # See e.g. https://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    # for the formula
    return -2 + 2 * np.cos(np.arange(1, n+1) * np.pi / (n + 1))


def laplace(n, dtype=None):
    """ Create a Laplacian operator of dimension n.

    The Laplacian operator is the symmetric, tridiagonal matrix with -2 in the
    main diagonal, and 1 in the upper/lower diagonal.
    """
    lower = np.ones(n-1, dtype=dtype)
    data = [-2 * np.ones(n, dtype=dtype), lower, lower]
    return sp.diags_array(data, offsets=[0, -1, 1])
