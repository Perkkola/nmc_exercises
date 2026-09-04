import numpy as np
from numpy.typing import NDArray


def naive_schur_complement(A: NDArray):
    n = A.shape[0] // 2
    # Partition the matrix
    A_11 = A[:n, :n]
    A_12 = A[:n, n:]
    A_21 = A[n:, :n]
    A_22 = A[n:, n:]

    # Compute the inverse of A_11
    A_11_inv = np.linalg.inv(A_11)

    # Compute the Schur complement
    S = A_22 - A_21 @ A_11_inv @ A_12

    return S

if __name__ == "__main__":
    n = 4
    # Generate random invertible matrix
    A = np.random.rand(n, n)
    mx = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, mx)

    S = naive_schur_complement(A)
    print(S)
