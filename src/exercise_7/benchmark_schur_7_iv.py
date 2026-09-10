import time

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from exercise_7.make_schur_system import make_schur_system
from exercise_7.naive_schur_complement_7_i import naive_schur_complement
from exercise_7.linear_eq_schur_complement_7_ii import linear_eq_schur_complement
from exercise_7.partial_gauss_schur_complement_7_iii import partial_gauss_schur_complement

def scipy_partial_gauss_schur_complement(A: NDArray, n1: int):
    n = A.shape[0]
    n1 = n // 2 if n1 is None else n1

    A_11 = A[:n1, :n1]
    A_12 = A[:n1, n1:]
    A_21 = A[n1:, :n1]
    A_22 = A[n1:, n1:]
    lu, piv = lu_factor(A_11)
    X = lu_solve((lu, piv), A_21.T, trans=1)
    M = X.T
    return A_22 - M @ A_12

METHODS = [
    ("naive (inv)", naive_schur_complement),
    ("linear eq (solve)", linear_eq_schur_complement),
    ("partial Gauss (LU)", partial_gauss_schur_complement),
    ("scipy partial Gauss (LU)", scipy_partial_gauss_schur_complement),
]


def time_method(fn, A, n1, repeats=3):
    best = np.inf
    for _ in range(repeats):
        A_copy = A.copy()
        t0 = time.perf_counter()
        fn(A_copy, n1)
        best = min(best, time.perf_counter() - t0)
    return best


def main(grid_sizes=(10, 20, 30, 40, 50), repeats=3):
    header = f"{'n':>4} {'n1':>6} " + " ".join(
        f"{name:>20}" for name, _ in METHODS
    )
    print(header)
    print("-" * len(header))

    for n in grid_sizes:
        A11, _, _, _, K = make_schur_system(n, dense=True)
        n1 = A11.shape[0]

        times = []
        for _, fn in METHODS:
            if n > 30 and fn.__name__ == "partial_gauss_schur_complement": 
                dt = np.nan  # too slow for large n
            else:
                dt = time_method(fn, K, n1, repeats)
            times.append(dt)

        print(f"{n:>4} {n1:>6} "
              + " ".join(f"{dt * 1e3:>17.3f} ms" for dt in times))


if __name__ == "__main__":
    main()
