import numpy as np
from numpy.typing import NDArray

def linear_eq_schur_complement(A: NDArray, n1: int | None = None):
    n = A.shape[0]
    n1 = n // 2 if n1 is None else n1
    # Partition the matrix
    A_11 = A[:n1, :n1]
    A_12 = A[:n1, n1:]
    A_21 = A[n1:, :n1]
    A_22 = A[n1:, n1:]

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
