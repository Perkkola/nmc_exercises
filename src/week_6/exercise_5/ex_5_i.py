import numpy as np
from numpy.typing import NDArray

def my_arnoldi(A: NDArray, b: NDArray, N: int):
    """The Matlab code for my_arnoldi translated to Python"""

    n = A.shape[0]
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    q = b

    terminated_at = 0 # I track the number of iterations so I can use it to slice the subspace later
    for i in range(N):
        for k in range(i):
            R[k, i] = np.vdot(q, Q[:, k])
            q = q - R[k, i] * Q[:, k]
        norm = np.linalg.norm(q)

        if norm < 1e-9: 
            break # Added a check for termination

        R[i, i] = norm
        Q[:, i] = q / norm
        q = A @ Q[:, i]
        terminated_at += 1

    return Q, R, terminated_at # Return the termination index also

if __name__ == "__main__":
    n = 20
    b = np.random.rand(20)
    Q, R = np.linalg.qr(np.random.rand(n, n))
    diag = np.diag(np.random.rand(n))

    A = Q @ diag @ Q.T

    V, _, j = my_arnoldi(A, b, n)
    V_i = V[:, :j]

    x_i = V @  np.linalg.inv( V_i.T @ A @  V_i) @  V_i.T @ b # (2.48)

    print(A @ x_i)
    print(b)
