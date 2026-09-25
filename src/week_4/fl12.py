import numpy as np
from numpy.typing import NDArray

def fp_round(X: NDArray):
    # Reshape the array so we can accept scalars and 1 dimensional arrays as input
    X_original_shape = X.shape
    X_0 = X.shape[0] if len(X.shape) >= 1 else 0
    X_1 = X.shape[1] if len(X.shape) >= 2  else 0
    X = X.reshape((max(1, X_0), max(1, X_1)))

    nM = 12
    Y = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]

            if x == 0:
                Y[i, j] = 0
                continue

            s = x / abs(x)
            x = abs(x)

            n = int(np.floor(np.log2(x)))

            y = 0

            for k in range(nM):
                rem = x % (2 ** (n - k))

                if np.abs(rem - x) > 1e-9: y += 2 ** (n - k)
                x = rem

            Y[i, j] = s * y

    return Y.reshape(X_original_shape)


if __name__ == "__main__":
    A = np.random.rand(4, 4)
    A_f12 = fp_round(A)

    print(A)
    print(A_f12)