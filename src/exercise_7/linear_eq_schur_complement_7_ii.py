import numpy as np
from numpy.typing import NDArray
from exercise_7.naive_schur_complement_7_i import naive_schur_complement

def linear_eq_schur_complement(A: NDArray):
    n = A.shape[0] // 2
    # Partition the matrix
    A_11 = A[:n, :n]
    A_12 = A[:n, n:]
    A_21 = A[n:, :n]
    A_22 = A[n:, n:]

    # Compute A_11^{-1} @ A_12
    sol = np.linalg.solve(A_11, A_12)

    # Compute the Schur complement
    S = A_22 - A_21 @ sol
    return S

if __name__ == "__main__":
    n = 6
    # Generate random invertible matrix
    A = np.random.rand(n, n)
    mx = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, mx)

    S = linear_eq_schur_complement(A)
