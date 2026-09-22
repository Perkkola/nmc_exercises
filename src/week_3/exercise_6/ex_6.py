import numpy as np
from numpy.typing import NDArray
from week_3.exercise_5.ex_5 import base_2_approximation

def appr_chol(A: NDArray, f: int = 10):
    n = len(A)

    if n == 1: return base_2_approximation(np.sqrt(A), f)

    L = np.zeros(A.shape)

    # I'm not sure if I'm supposed to apply the base 2 approximation to these also but I decided against it since
    # in my mind these are not "operations".
    a_11 = A[0, 0]
    a_21 = A[1:, 0]
    A_22 = A[1:, 1:]

    # use the approximation from the previous exercise
    base_2_sqrt_a_11 = base_2_approximation(np.sqrt(a_11), f)
    L[0, 0] = base_2_sqrt_a_11
    L[1:, 0] = base_2_approximation(a_21 / base_2_sqrt_a_11, f)

    outer = base_2_approximation(np.outer(a_21, a_21.T), f)
    outer_div_a11 = base_2_approximation(outer / a_11, f)
    S = base_2_approximation(A_22 - outer_div_a11, f)
    L[1:, 1:] = appr_chol(S)

    return L

if __name__ == "__main__":
    n = 100

    err = 0
    for _ in range(n):
        X = np.random.rand(10, 1)
        A = np.outer(X.T, X) + np.eye(10)

        L = appr_chol(A)

        err += np.linalg.norm(np.abs(A - L @ L.T))

    print(f"Mean error (Frobenius norm difference) over {n} iterations: {err / n}")
    print(f"Mean accuracy of base 2 approximation with 10 significant figures: {1 - err / n}")