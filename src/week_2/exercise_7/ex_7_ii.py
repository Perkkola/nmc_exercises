import numpy as np
import sys
from numpy.typing import NDArray
from week_2.exercise_7.ex_7_i import generate_s_p_s_d

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def rank_revealing_cholesky(A: NDArray):
    n = len(A)
    if n == 1: return np.eye(1), np.sqrt(A[0, 0]) #Base case
    
    # Find the index of the larges diagonal value
    i = np.argmax(A.diagonal())

    # Construct the permutation matrix
    P = np.eye(n)
    P[0, 0] = 0
    P[i, i] = 0
    P[0, i] = 1
    P[i, 0] = 1

    # Pivot largest diagonal element to (0, 0)
    A_tilde = P.T @ A @ P

    if A_tilde[0, 0] <= 1e-10: return np.eye(n), np.zeros((n, n)) # Diagonal is zero. Tolerance for numerical stability

    a_tilde_11 = A_tilde[0, 0]
    a_tilde_21 = A_tilde[1:, 0]
    A_tilde_22 = A_tilde[1:, 1:]

    # Recurse with a submatrix
    P_2, L_2 = rank_revealing_cholesky(A_tilde_22 - np.outer(a_tilde_21, a_tilde_21.T) / a_tilde_11)

    # Collect the permutation matrices and the lower triangular matrices
    return P @ np.block([[1, np.zeros((1, n - 1))],
                         [ np.zeros((n - 1, 1)), P_2]]), np.block([[np.sqrt(a_tilde_11), np.zeros((1, n - 1))],
                                                                   [ (P_2.T @ a_tilde_21 / np.sqrt(a_tilde_11))[:, np.newaxis], L_2]])

if __name__ == "__main__":
    rand_s_p_s_d = generate_s_p_s_d(5, 2)
    P, L = rank_revealing_cholesky(rand_s_p_s_d)

    print(rand_s_p_s_d)
    print(P @ L @ L.T @ P.T)
