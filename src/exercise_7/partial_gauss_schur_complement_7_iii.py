import numpy as np
from scipy.linalg import lu_solve, lu_factor 
from numpy.typing import NDArray
from exercise_7.naive_schur_complement_7_i import naive_schur_complement
from exercise_7.linear_eq_schur_complement_7_ii import linear_eq_schur_complement
from exercise_6.lu2_6_ii import lu_decomposition

def partial_gauss_schur_complement(A: NDArray):
    n = A.shape[0] // 2

    A_11 = A[:n, :n]
    A_12 = A[:n, n:]
    A_21 = A[n:, :n]
    A_22 = A[n:, n:]

    p, L, U = lu_decomposition(A_11.copy(), A_11.shape[0])

    lu = L - np.eye(n) + U
    # Solve for A11.T @ X = A21.T using scipy LU solve
    X = lu_solve((lu, p), A_21.T, trans=1)
    M = X.T         # M = A21 @ inv(A11)

    S = A_22 - M @ A_12 

    return S

if __name__ == "__main__":
    n = 4
    # Generate random invertible matrix
    A = np.random.rand(n, n)
    mx = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, mx)

    A_naive = A.copy()
    A_linear_eq = A.copy()
    S = partial_gauss_schur_complement(A)