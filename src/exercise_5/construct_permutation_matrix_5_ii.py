import numpy as np
from exercise_5.generate_random_morally_triangular_matrices_5_i import generate_random_morally_triangular_matrices
from numpy.typing import NDArray

def construct_permutation_matrix(A: NDArray, n: int, offset: int = 0):
    if offset == n - 1: return np.eye(n) # Base case

    # Find the index of the row we want to swap to the top
    smallest_row_index = n  - 1
    smallest_column_index = n  - 1
    for index, row in enumerate(A[offset:]):
        first_non_zero_column_index = np.nonzero(row)[0][0] if not np.nonzero(row)[0].size == 0 else n  - 1 # Check if the row is all zeros
        if first_non_zero_column_index < smallest_column_index: 
            smallest_row_index = index
            smallest_column_index = first_non_zero_column_index

    # Adjust the index according to the current recursion level
    smallest_row_index += offset

    # Construct the permutation matrix
    P = np.eye(n)
    P[offset, offset] = 0
    P[smallest_row_index, smallest_row_index] = 0
    P[offset, smallest_row_index] = 1
    P[smallest_row_index, offset] = 1

    # Recurse. A permutation matrix times a permutation matrix is a permutation matrix.
    return P @ construct_permutation_matrix(P @ A, n, offset + 1)

if __name__ == "__main__":
    n = 4
    A = generate_random_morally_triangular_matrices(n)
    print(A)
    P = construct_permutation_matrix(A, n)
    print(P.T @ A)