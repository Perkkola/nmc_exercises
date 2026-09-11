import numpy as np
from numpy.typing import NDArray


def rchol(A: NDArray):
    n = len(A)
    if n == 1: return np.sqrt(A)

    a_11 = A[0, 0]
    a_21 = A[1:, 0]
    A_22 = A[1:, 1:]

    print(a_11)

    return np.block([[np.sqrt(a_11), np.zeros((1, n - 1))],
                  [(a_21 / np.sqrt(a_11))[:, np.newaxis], rchol(A_22 - (a_21 @ a_21.T) / a_11)]])

n = 5
F = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        F[i, j] = np.random.rand() + 1


L = rchol(F.T @ F)
print(L)