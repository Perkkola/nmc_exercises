import numpy as np
from numpy.typing import NDArray

def arnoldi(A: NDArray, b: NDArray):
    Q = []
    Q.append(b)

    n = len(A)
    for i in range(1, n):
        q = A @ Q[i - 1]
        for k in range(i):
            q = q - q.T @ Q[k] @ Q[k]
        Q.append(q / np.linalg.norm(q))
        
    return Q.T

if __name__ == "__main__":
    L = [1, 10, 100, 1000]
    for l in L:
        for i in range(5, 15):
            n = 20
            Q, R = np.linalg.qr(np.random.rand(n, n))
            diag = (np.random.rand(1, n) + 1) * L
            A = Q @ diag @ Q.T