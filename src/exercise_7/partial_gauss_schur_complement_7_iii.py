import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from exercise_6.lu2_6_ii import lu_decomposition


def partial_gauss_schur_complement(A: NDArray, n1: int | None = None):
    n = A.shape[0]
    n1 = n // 2 if n1 is None else n1

    A_11 = A[:n1, :n1]
    A_12 = A[:n1, n1:]
    A_21 = A[n1:, :n1]
    A_22 = A[n1:, n1:]

    p, L_11, U_11 = lu_decomposition(A_11.copy(), n1)
    q = np.argsort(p)

    # Forward substitution: C12 = L11^{-1}
    C_12 = solve_triangular(L_11, A_12[q, :], lower=True, unit_diagonal=True)

    # Back substitution on the transposed system. Solve U11^T W^T = A_21^T, so W = A_21 U11^{-1}
    W = solve_triangular(U_11, A_21.T, lower=False, trans="T").T

    S = A_22 - W @ C_12
    return S


if __name__ == "__main__":
    n = 6
    # Generate random invertible matrix
    A = np.random.rand(n, n)
    mx = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, mx)

    S = partial_gauss_schur_complement(A)