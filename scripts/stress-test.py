"""
Script to benchmark SLEPCs, ARPACK and our implementations across different set
of parameters for a given matrix.

See the script plot-stress-test.py about plotting the results.
"""
import argparse
import csv
import os.path
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from utils import (
    WHICH_TO_SORT, ConvergenceTracker, EigensolverParameters, Statistics,
    arnoldi_py_eig, arpack_eig, arnoldi_py_eig, slepc_eig, find_best_matching,
    load_suitesparse_mat, print_residuals
)


TOL = 1e-8
MAX_RESTARTS = 100_000
WHICH = "LR"

PARAMETERS = []
for WHICH in ["LM", "LR"]:
    PARAMETERS.extend([
        EigensolverParameters(3, 20, TOL, MAX_RESTARTS, 10, WHICH),
        EigensolverParameters(10, 20, TOL, MAX_RESTARTS, 16, WHICH),
        EigensolverParameters(6, 20, TOL, MAX_RESTARTS, 12, WHICH),
        EigensolverParameters(12, 30, TOL, MAX_RESTARTS, 21, WHICH),
        EigensolverParameters(20, 40, TOL, MAX_RESTARTS, 30, WHICH),
        EigensolverParameters(30, 50, TOL, MAX_RESTARTS, 40, WHICH),
        EigensolverParameters(50, 80, TOL, MAX_RESTARTS, 40, WHICH),
    ])

def main():
    parser = argparse.ArgumentParser(
        description="Compare partial_schur against ARPACK on a SuiteSparse matrix."
    )
    parser.add_argument("mat_file", help="Path to the .mat file (SuiteSparse format)")
    parser.add_argument("-o", "--output-path", help="CSV Out path", default="output.csv")

    args = parser.parse_args()

    A_raw = load_suitesparse_mat(args.mat_file)
    n = A_raw.shape[0]
    nnz = A_raw.nnz
    A = A_raw.astype(np.complex128)
    print(f"Matrix: {args.mat_file}")
    print(f"  shape={n}x{n}, nnz={nnz}, dtype={A.dtype}")

    tracker = ConvergenceTracker()

    with open(args.output_path, "wt") as fp:
        fieldnames = ["method", "dtype", "nev", "ncv", "tol", "max_restarts",
                      "p", "which", "elapsed", "matvecs", "restarts",
                      "match"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for parameters in PARAMETERS:
            print(parameters)
            arpack_vals, arpack_vecs, arpack_stats = arpack_eig(A, parameters)
            ps_vals, ps_vecs, ps_stats = arnoldi_py_eig(A, parameters)
            slepc_vals, slepc_vecs, slepc_stats = slepc_eig(A, parameters, tracker)

            print(f"\n--- Perf comparison ---")
            print(f"  ARPACK:        {arpack_stats.matvecs} matvecs in {arpack_stats.restarts} iterations  ({arpack_stats.elapsed:.2f}s)")
            print(f"  partial_schur: {ps_stats.matvecs} matvecs in {ps_stats.restarts} iterations  ({ps_stats.elapsed:.2f}s)")
            print(f"  SLEPC:         {slepc_stats.matvecs} matvecs in {slepc_stats.restarts} iterations  ({slepc_stats.elapsed:.2f}s)")

            x, y = find_best_matching(arpack_vals, ps_vals)
            try:
                np.testing.assert_allclose(y, x, rtol=parameters.tol)
                match = True
            except AssertionError as e:
                match = False
                print("\033[31m!!! ARPACK and Krylov-Schur don't match !!!\033[0m")
                print(e)

            print_residuals("ARPACK", A, arpack_vals, arpack_vecs)
            print_residuals("Krylov-Schur", A, ps_vals, ps_vecs)

            for method, stats in zip(["arpack", "krylov-schur", "slepc"], [arpack_stats, ps_stats, slepc_stats]):
                row = {
                    "method": method,
                    "dtype": stats.dtype,
                    "nev": parameters.nev,
                    "ncv": parameters.ncv,
                    "tol": parameters.tol,
                    "max_restarts": parameters.max_restarts,
                    "p": parameters.p,
                    "which": parameters.which,
                    "elapsed": stats.elapsed,
                    "matvecs": stats.matvecs,
                    "restarts": stats.restarts,
                    "match": match,
                }
                writer.writerow(row)

if __name__ == "__main__":
    main()
