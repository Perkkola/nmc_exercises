import numpy as np
from week_6.exercise_5.ex_5_i import my_arnoldi
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

# This function implements the A-norm
def error(x, x_i, A):
    diff = x - x_i
    return np.sqrt(diff.T @ A @ diff)

if __name__ == "__main__":
    n = 20
    b = np.random.rand(20)

    plt.figure()
    for L in [1.0, 10.0, 100.0, 1000.0]:
        Q, R = np.linalg.qr(np.random.rand(n, n))
        A = Q @ np.diag(np.linspace(1.0, L, n)) @ Q.T

        x = np.linalg.solve(A, b) # Exact solution for reference

        errors_for_L = []

        for i in range(5, 16):
            V, _, j = my_arnoldi(A, b, i) # Use the i-th approximation

            V_i = V[:, :j] # Slice only until j, otherwise V_i is singular
            x_i = V_i @ np.linalg.inv(V_i.T @ A @ V_i) @ V_i.T @ b # (2.48)

            err = error(x, x_i, A)
            
            errors_for_L.append(err)

        plt.semilogy(list(range(5, 16)), errors_for_L, label=f"L = {int(L)}")

    plt.title("Error decay for different L and i")
    plt.xlabel("i")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()
