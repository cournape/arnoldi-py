"""
Script to benchmark SLEPCs, ARPACK and our implementations across different set
of parameters for a given matrix.

See the script plot-stress-test.py about plotting the results.
"""
import argparse
import csv
import sys

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from utils import (
    WHICH_TO_SORT, ConvergenceTracker, EigensolverParameters,
    arnoldi_py_eig, arpack_eig, slepc_eig, find_best_matching,
    load_suitesparse_mat, print_residuals
)


TOL = 1e-8
MAX_RESTARTS = 100_000

def main():
    parser = argparse.ArgumentParser(
        description="Compare partial_schur against ARPACK on a SuiteSparse matrix."
    )
    parser.add_argument("mat_file", help="Path to the .mat file (SuiteSparse format)")
    parser.add_argument("-o", "--output-path", help="CSV Out path", default=None)
    parser.add_argument("-p", "--parameters-path", help="CSV of parameters", default=None)

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = Path(args.mat_file).with_suffix(".csv")

    if args.parameters_path is None:
        parameters_list = []
        for which in ["LM", "LR"]:
            parameters_list.extend([
                EigensolverParameters(10, 20, TOL, MAX_RESTARTS, 16, which),
                EigensolverParameters(12, 30, TOL, MAX_RESTARTS, 21, which),
                EigensolverParameters(20, 40, TOL, MAX_RESTARTS, 30, which),
                EigensolverParameters(30, 50, TOL, MAX_RESTARTS, 40, which),
                EigensolverParameters(35, 80, TOL, MAX_RESTARTS, 60, which),
                EigensolverParameters(45, 100, TOL, MAX_RESTARTS, 70, which),
            ])
    else:
        def decomment(fp):
            for line in fp:
                if not line.startswith("#"):
                    yield line

        with open(args.parameters_path, "rt", newline="") as fp:
            reader = csv.DictReader(decomment(fp))
            parameters_list = [
                EigensolverParameters(
                    int(d["nev"]),
                    int(d["ncv"]),
                    float(d["tol"]),
                    int(d["max_restarts"]),
                    int(d["p"]),
                    d["which"],
                )
                for d in reader
            ]

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

        for parameters in parameters_list:
            print(parameters)
            print("Runing ARPACK ...")
            arpack_vals, arpack_vecs, arpack_stats = arpack_eig(A, parameters)
            print("Runing Krylov-Schur ...")
            ps_vals, ps_vecs, ps_stats = arnoldi_py_eig(A, parameters)
            print("Runing SLEPc ...")
            slepc_vals, slepc_vecs, slepc_stats = slepc_eig(A, parameters, tracker)

            print(f"\n--- Perf comparison ---")
            print(f"  ARPACK:        {arpack_stats.matvecs} matvecs in {arpack_stats.restarts} iterations  ({arpack_stats.elapsed:.2f}s)")
            print(f"  partial_schur: {ps_stats.matvecs} matvecs in {ps_stats.restarts} iterations  ({ps_stats.elapsed:.2f}s)")
            print(f"  SLEPC:         {slepc_stats.matvecs} matvecs in {slepc_stats.restarts} iterations  ({slepc_stats.elapsed:.2f}s)")
            print(f"  SLEPc call counts:")
            print(f"    MatMult:              {slepc_stats.count_matvec}")
            print(f"    STApply (A@x):        {slepc_stats.count_st_apply}")
            print(f"    BVOrthogonalizeCol:   {slepc_stats.count_ortho}")
            print(f"    BVDotVec (V^H@w):     {slepc_stats.count_dot}")
            print(f"    BVMultVec (w-=V*c):   {slepc_stats.count_multivec}")
            print(f"    DSSolve (restart):    {slepc_stats.count_ds_solve}")

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
