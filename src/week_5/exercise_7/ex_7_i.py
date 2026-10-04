import numpy as np
import matplotlib.pyplot as plt

def cg(A, b, x_0, tol = 1e-9, x = None):
    P = []
    errors = []
    r = b - A @ x_0 # (2.20)
    p = r.copy()

    P.append(p)
    while np.linalg.norm(r) > tol:
        denominator = (p @ A @ p)

        if denominator == 0:
            break

        alpha = p @ r / denominator # (2.34)
        x_0 = x_0 + alpha * p # (2.35)

        e = error(x, x_0, A) # Compute the error for each approximation
        errors.append(e)

        r = r - alpha * A @ p # (2.36)
        p = r.copy()
        for p_prev in P:
            p -= (r @ A @ p_prev) / (p_prev @ A @ p_prev) * p_prev # (2.32)

        P.append(p)

    return x_0, errors

# This function implements the A-norm
def error(x, x_i, A):
    diff = x - x_i
    return np.sqrt(diff.T @ A @ diff)

if __name__ == "__main__":
    t_s = [0.1, 0.5, 1, 5, 10]
    b = np.array([1, 2])
    x_0 = np.array([0, 0])

    errors = []

    for t in t_s:
        Q = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        D_A = np.diag([1, t])

        A = Q.T @ D_A @ Q

        x = np.linalg.solve(A, b) # Solve x exactly for reference.

        x_i, errs = cg(A, b, x_0, x = x)
        

        errors.append(errs)

    fig, ax = plt.subplots()
    ax.set_title("Error vs. Number of iterations for different values of t")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Error")

    for t, error_for_i in zip(t_s, errors):
        ax.plot(range(1, len(error_for_i) + 1), error_for_i, label=f"t = {t}")

    ax.set_yscale("log")
    ax.legend()
    plt.show()
