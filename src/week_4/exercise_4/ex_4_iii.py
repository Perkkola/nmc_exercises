import numpy as np
from week_4.exercise_4.ex_4_i import myprod12
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ratios = []
    bounds = []
    max_N = 35

    for n in range(2, max_N):
        machine_epsilon = 1 / (2 ** 11)
        bound = n * machine_epsilon / (1 - n * machine_epsilon)

        bounds.append(bound)
        s = 0

        # Take the mean over 1000 iterations
        num = 1000
        for _ in range(num):
            x = np.random.rand(1, n)
            y = np.random.rand(n, 1)

            ratio = np.abs(myprod12(x, y)[0, 0] - (x @ y)[0, 0]) / (np.linalg.norm(x) * np.linalg.norm(y))
            s += ratio

        ratios.append(s / num)

    # Plot the results using matplotlib
    N = max_N - 2

    ind = np.arange(N)

    plt.figure(figsize=(10,5))

    width = 0.3       

    plt.bar(ind, ratios , width, label='Ratio')
    plt.bar(ind + width, bounds, width, label='Bound')

    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title('Comparison of the ratio vs. the bound for different values of n')

    plt.xticks(ind + width / 2, [str(x) for x in range(2, max_N)])

    plt.legend(loc='best')
    plt.show()

