import numpy as np
from numpy.typing import NDArray
from week_4.fl12 import fp_round

def myprod12(A: NDArray, B: NDArray):
    assert A.shape[1] == B.shape[0]
    prod = np.empty((A.shape[0], B.shape[1]))

    for i in range(B.shape[1]):
        for j in range(A.shape[0]):
            y = 0

            for k in range(A.shape[1]):
                y = fp_round(y + fp_round(A[j, k] * B[k, i]))

            prod[j, i] = y

    return prod

if __name__ == "__main__":
    B = np.random.rand(4, 4)
    A = np.random.rand(4, 4)
    print(A @ B)
    print(myprod12(A, B))