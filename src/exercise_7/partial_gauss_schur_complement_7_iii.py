import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from exercise_6.lu2_6_ii import lu_decomposition


def partial_gauss_schur_complement(A: NDArray, n1: int | None = None):
    """Schur complement S = A22 - A21 A11^{-1} A12 by partial Gaussian elimination.

    Follows problem 3(iii): with A11 = L11 U11,

        A = [[L11,  0], [[U11, C12],
             [W,    I]]  [0,   S  ]]

    where C12 = L11^{-1} A12 and W = A21 U11^{-1}, so S = A22 - W C12.
    Only triangular solves are needed; A11^{-1} is never formed.

    n1 is the size of the A11 block (defaults to A.shape[0] // 2).
    """
    n = A.shape[0]
    n1 = n // 2 if n1 is None else n1

    A_11 = A[:n1, :n1]
    A_12 = A[:n1, n1:]
    A_21 = A[n1:, :n1]
    A_22 = A[n1:, n1:]

    # Your lu2 returns p such that (L11 @ U11)[p, :] = A_11,
    # equivalently A_11[q, :] = L11 @ U11 with q = argsort(p).
    p, L11, U11 = lu_decomposition(A_11.copy(), n1)
    q = np.argsort(p)

    # Forward substitution: C12 = L11^{-1} (A_11 rows permuted the same way => A_12 rows too)
    C12 = solve_triangular(L11, A_12[q, :], lower=True, unit_diagonal=True)

    # Back substitution on the transposed system: solve U11^T W^T = A_21^T, so W = A_21 U11^{-1}
    W = solve_triangular(U11, A_21.T, lower=False, trans="T").T

    S = A_22 - W @ C12
    return S


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def reference(A, n1):
        return A[n1:, n1:] - A[n1:, :n1] @ np.linalg.solve(A[:n1, :n1], A[:n1, n1:])

    # 1) Random matrix, no row swaps needed.
    n, n1 = 12, 7
    A = rng.random((n, n))
    print("no pivoting     :", np.allclose(partial_gauss_schur_complement(A, n1), reference(A, n1)))

    # 2) Force zero pivots in A11 so lu2 actually swaps rows.
    B = rng.random((n, n))
    B[0, 0] = 0.0
    B[1, :2] = 0.0
    print("with row swaps  :", np.allclose(partial_gauss_schur_complement(B, n1), reference(B, n1)))

    # 3) Uneven split as in the FEM system.
    print("uneven split    :", np.allclose(partial_gauss_schur_complement(B, 3), reference(B, 3)))