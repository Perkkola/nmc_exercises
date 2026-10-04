import numpy as np
from week_5.exercise_6.ex_6_i import gsA

if __name__ == "__main__":
    n = 4
    k = 3

    F = np.random.rand(n, n)
    A = F @ F.T

    X = np.random.rand(n, k)

    P = gsA(A, X)

    P = P.T
    for i in range(P.shape[0]):
        assert np.abs(1 - np.sqrt(P[i] @ A @ P[i])) < 1e-9, f"Vector {i} is not normalized"

    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if i != j: assert np.abs(np.dot(P[i], A @ P[j])) < 1e-9, f"Vectors {i} and {j} are not orthogonal"

    # also check R(P) == R(X)
    P_orig = P.T
    rank_P = np.linalg.matrix_rank(P)
    rank_X = np.linalg.matrix_rank(X)

    assert rank_P == rank_X, f"R(X) is not equal to R(P)"

    print("All tests passed!")