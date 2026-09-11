import numpy as np
import cmath
from numpy.typing import NDArray
import sys

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def cholesky_with_update_strategy(A: NDArray):
    n = len(A)
    if n == 1: return cmath.sqrt(A[0, 0]) 

    A_tilde = A
    L = np.zeros((n, n), dtype="complex")

    # Following Jaakko's hint given at the exercise session.
    for i in range(n - 1):
        a_tilde_11 = A_tilde[0, 0]
        a_tilde_21 = A_tilde[1:, 0]
        A_tilde_22 = A_tilde[1:, 1:]

        L[i, i] = cmath.sqrt(a_tilde_11)
        L[(i + 1):, i] = a_tilde_21 / cmath.sqrt(a_tilde_11)

        A_tilde = A_tilde_22 - np.outer(a_tilde_21, a_tilde_21.T) / a_tilde_11

    # Update righmost corner
    L[n - 1, n - 1] = cmath.sqrt(A_tilde[0][0])

    return L

if __name__ == "__main__":
    n = 5
    F = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            F[i, j] = np.random.rand()

    L = cholesky_with_update_strategy(F.T @ F)
    print(*L, sep="\n")