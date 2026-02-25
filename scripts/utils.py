import numpy as np
import scipy.io
import scipy.sparse as sp

from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator

from arnoldi.utils import arg_largest_real, arg_largest_magnitude


WHICH_TO_SORT = {
    "LM": arg_largest_magnitude,
    "LR": arg_largest_real,
}


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
