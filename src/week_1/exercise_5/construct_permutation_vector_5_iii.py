import numpy as np
from src.week_1.exercise_5.generate_random_morally_triangular_matrices_5_i import generate_random_morally_triangular_matrices
from numpy.typing import NDArray

# Permutes the indices of a list in place using XOR
def permute_indices(p: NDArray, indices: tuple):
    if indices[0] == indices[1]: return p
    p[indices[0]] ^= p[indices[1]]
    p[indices[1]] ^= p[indices[0]]
    p[indices[0]] ^= p[indices[1]]
    return p

def construct_permutation_vector(A: NDArray, n: int, offset: int = 0):
    if offset == n - 1: return [x for x in range(n)] # Base case is the unpermuted indices

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

    # Permute the rows of A according to the found indices
    p = [offset, smallest_row_index]
    A = A.copy()
    A[p] = A[list(reversed(p))]

    return permute_indices(construct_permutation_vector(A, n, offset + 1), (offset, smallest_row_index))

if __name__ == "__main__":
    n = 4
    A = generate_random_morally_triangular_matrices(n)
    print(A)
    p = construct_permutation_vector(A, n)
    print(p)
    A[p] = A[[x for x in range(4)]]
    print(A)