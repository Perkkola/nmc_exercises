import numpy as np
from week_4.exercise_4.ex_4_i import myprod12
import matplotlib.pyplot as plt

if __name__ == "__main__":
    max_errors = []
    bounds = []
    max_N = 21
    num = 100

    for n in range(2, max_N):
        machine_epsilon = 1 / (2 ** 11)
        bound = n * machine_epsilon / (1 - n * machine_epsilon)

        bounds.append(bound)
        s = 0

        # Take the mean over 1000 iterations
        for _ in range(num):
            U, _ = np.linalg.qr(np.random.rand(n, n))
            V, _ = np.linalg.qr(np.random.rand(n, n))

            max_error = np.matrix.max(np.matrix(np.abs(myprod12(U, V) - U @ V)))
            s += max_error

        max_errors.append(s / num)
    # Plot the results using matplotlib
    N = max_N - 2

    ind = np.arange(N)

    plt.figure(figsize=(10,5))

    width = 0.3       

    plt.bar(ind, max_errors , width, label='Max absolute error')
    plt.bar(ind + width, bounds, width, label='Bound')

    plt.xlabel('n')
    plt.ylabel('Error')
    plt.title('Comparison of the max absolute error vs. the bound for different values of n')

    plt.xticks(ind + width / 2, [str(x) for x in range(2, max_N)])

    plt.legend(loc='best')
    plt.show()

