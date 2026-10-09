import numpy as np

def gsA(A, X):
    X = X.T
    P = np.zeros((X.shape[0], X.shape[1]))
    p_0 = X[0] / np.sqrt(X[0] @ A @ X[0]) # Normalize using A-norm
    P[0] = p_0

    for i in range(1, X.shape[0]):
        a = X[i].copy()
        for j in range(i):
            a -= (a @ A @ P[j]) * P[j] # Remove the A-inner product component using each p_j
        P[i] = a / np.sqrt(a @ A @ a) # Normalize using A-norm

    return P.T

if __name__ == "__main__":
    n = 4
    k = 3

    F = np.random.rand(n, n)
    A = F @ F.T

    X = np.random.rand(n, k)

    P = gsA(A, X)

