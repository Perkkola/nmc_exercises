import numpy as np
import cmath
from week_2.exercise_6.ex_6_i import rchol
from week_2.exercise_6.ex_6_ii import cholesky_with_update_strategy

if __name__ == "__main__":
    # Test for 100 random matrices of sizes n=1..10
    for n in range(1, 11):
        for _ in range(100):
            F = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    F[i, j] = np.random.rand()

            A = F.T @ F
            L_update = cholesky_with_update_strategy(A.copy())
            L_rchol = rchol(A.copy())

            # For n == 1, A is a matrix but L_update and L_rchol are scalars
            if len(A) == 1:
                assert cmath.isclose(A[0][0], L_update ** 2), "Reconstruction failed for the update method"
                assert cmath.isclose(A[0][0], L_rchol ** 2), "Reconstruction failed for rchol"
            else:
                assert np.allclose(A, L_update @ L_update.T), "Reconstruction failed for the update method"
                assert np.allclose(A, L_rchol @ L_rchol.T), "Reconstruction failed for rchol"
                
    print("All tests passed!")
    