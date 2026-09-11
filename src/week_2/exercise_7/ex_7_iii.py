import numpy as np
from week_2.exercise_7.ex_7_i import generate_s_p_s_d
from week_2.exercise_7.ex_7_ii import rank_revealing_cholesky

if __name__ == "__main__":
    n = 5
    k = 3

    # Generate random s.p.s.d matrix using our function
    rand_s_p_s_d = generate_s_p_s_d(n, k)
    P, L = rank_revealing_cholesky(rand_s_p_s_d)

    # Get the error of the factorization as the Frobenius norm of the absolute difference
    error = np.linalg.norm(np.abs(rand_s_p_s_d - P @ L @ L.T @ P.T))
    print(f"Reconstruction error: {error}")

    # Accuracy is then 1 - error
    print(f"Factorization accuracy: {1 - error}")