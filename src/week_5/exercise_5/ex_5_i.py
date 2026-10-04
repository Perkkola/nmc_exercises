import numpy as np
import matplotlib.pyplot as plt

def line_search(A, b, x_0, i):
    for _ in range(i):
        p_i = b - A @ x_0 # (2.20)
        denominator = p_i.T @ A @ p_i 

        if denominator == 0:
            break

        alpha_i = (p_i.T @ p_i) / denominator # (2.21)
        x_0 = x_0 + alpha_i * p_i # (2.22)

    return x_0

# This function implements the A-norm
def error(x, x_i, A):
    diff = x - x_i
    return np.sqrt(diff.T @ A @ diff)[0][0]

if __name__ == "__main__":
    t_s = [0.1, 0.5, 1, 5, 10]
    i = 10
    b = np.array([[1], [2]])
    x_0 = np.array([[0], [0]])

    errors = []

    for t in t_s:
        Q = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        D_A = np.diag([1, t])

        A = Q.T @ D_A @ Q

        x = np.linalg.solve(A, b) # Solve x exactly for reference.
        errors_for_i = []

        for num_iter in range(i):

            x_i = line_search(A, b, x_0, num_iter)
            e = error(x, x_i, A) # Compute the error for each approximation
            errors_for_i.append(e)

        errors.append(errors_for_i)

    fig, ax = plt.subplots()
    ax.set_title("Error vs. Number of iterations for different values of t")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Error")

    for t, error_for_i in zip(t_s, errors):
        ax.plot(range(i), error_for_i, label=f"t = {t}")

    ax.set_yscale("log")
    ax.legend()
    plt.show()
