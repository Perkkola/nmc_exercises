import numpy as np
import matplotlib.pyplot as plt
from week_4.fl12 import fp_round

if __name__ == "__main__":
    two_norms = []
    inf_norms = []
    num = 1000
    max_n = 31
    for n in range(2, max_n):
        two_sum = 0
        inf_sum = 0
        for _ in range(num):
            U, _ = np.linalg.qr(np.random.rand(n, n))
            two_sum += np.linalg.norm(np.abs(U - fp_round(U)), 2)
            inf_sum += np.linalg.norm(np.abs(U - fp_round(U)), np.inf)

        two_norms.append(two_sum / num)
        inf_norms.append(inf_sum / num)

    # Plot the results using matplotlib
    N = max_n - 2

    ind = np.arange(N)

    plt.figure(figsize=(10,5))

    width = 0.3       

    plt.bar(ind, two_norms , width, label='2-norm')
    plt.bar(ind + width, inf_norms, width, label='∞-norm')

    plt.xlabel('n')
    plt.ylabel('Error')
    plt.title('Absolute error of the fl representation with 2 and ∞ -norms for different n')

    plt.xticks(ind + width / 2, [str(x) for x in range(2, max_n)])

    plt.legend(loc='best')
    plt.show()