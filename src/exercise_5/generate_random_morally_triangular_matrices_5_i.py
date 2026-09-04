import numpy as np

# Generate a upper triangular matrix with random values and then apply a random permutation
def generate_random_morally_triangular_matrices(n: int):
    A = []
    for i in range(n):
        row = np.zeros(n)
        for j in range(i, n):
            row[j] = np.random.rand()
        A.append(row)
    
    return np.random.permutation(A)
