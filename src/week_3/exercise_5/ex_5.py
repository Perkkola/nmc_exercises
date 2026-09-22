import numpy as np
from numpy.typing import NDArray
from week_3.exercise_4.ex_4_i import base_b

def base_2_approximation(X: NDArray, f: int):
    # I reshape the array so we can accept scalars and 1 dimensional arrays as input
    X_original_shape = X.shape
    X_0 = X.shape[0] if len(X.shape) >= 1 else 0
    X_1 = X.shape[1] if len(X.shape) >= 2  else 0
    X = X.reshape((max(1, X_0), max(1, X_1)))

    X_hat = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dec = X[i][j]
            sign = dec / abs(dec)
            ax = abs(dec)

            n = int(np.floor(np.log2(ax))) 
            mantissa = ax / (2.0 ** n)          # normalized into [1, 2)

            digits = base_b(2, mantissa, f)   # use the previous exercise to get the digits  
            value = sum(d * 2.0 ** (-k) for k, d in enumerate(digits)) # sum up the digits

            X_hat[i][j] = sign * value * (2.0 ** n) # reconstruct the decimal value using the formula from exericse 3

    # Reshape back to original shape
    return X_hat.reshape(X_original_shape)


if __name__ == "__main__":
    X = np.array([[123.123, 456.45678],
                  [67.69, 12.456]])

    print(base_2_approximation(X, 5))