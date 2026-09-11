import numpy as np
import cmath
from numpy.typing import NDArray
import sys

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def rchol(A: NDArray):
    n = len(A)
    if n == 1: return cmath.sqrt(A[0, 0]) # Base case

    # Select the vector and submatrix for the recursive step
    a_11 = A[0, 0]
    a_21 = A[1:, 0]
    A_22 = A[1:, 1:]

    # Recurse into the lower right submatrix
    return np.block([[cmath.sqrt(a_11), np.zeros((1, n - 1))],
                  [(a_21 / cmath.sqrt(a_11))[:, np.newaxis], rchol(A_22 - np.outer(a_21, a_21.T) / a_11)]])

if __name__ == "__main__":
    n = 5
    F = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            F[i, j] = np.random.rand()

    L = rchol(F.T @ F)
    print(*L, sep="\n")