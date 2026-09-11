import numpy as np
from numpy.typing import NDArray
from src.week_1.exercise_5.generate_random_morally_triangular_matrices_5_i import generate_random_morally_triangular_matrices

def permute_indices(p: NDArray, indices: tuple):
    if indices[0] == indices[1]: return p
    p[indices[0]] ^= p[indices[1]]
    p[indices[1]] ^= p[indices[0]]
    p[indices[0]] ^= p[indices[1]]
    return p

def lu_decomposition(A: NDArray, n: int, offset: int = 0):
    indices = (offset, offset)
    p = [*indices]
    if offset == n - 1: return [x for x in range(n)], np.eye(n), A # Base case
    if A[offset, offset] == 0:
        # Find the index of the row we want to swap to the top
        smallest_row_index = n  - 1
        smallest_column_index = n  - 1
        for index, row in enumerate(A[offset:]):
            assert np.nonzero(row)[0].size > 0, "Matrix is singular"
            first_non_zero_column_index = np.nonzero(row)[0][0]
            if first_non_zero_column_index < smallest_column_index: 
                smallest_row_index = index
                smallest_column_index = first_non_zero_column_index
    
        # Adjust the index according to the current recursion level
        smallest_row_index += offset
    
        # Construct the permutation vector
        indices = (offset, smallest_row_index)

    # Permute the rows of A according to the found indices
    p = [*indices]
    A = A.copy()
    A[p] = A[list(reversed(p))]

    # Get the vectors for the elimination matrix. Here we use the function 'permute_indices' instead of the permutation matrix
    a_11 = A[offset, offset]
    a_21 = A[offset + 1:, offset]

    active_size = n - offset
    E_full = np.eye(n)
    E_inv_full = np.eye(n)

    # The elimination matrix and its inverse as in the lecture notes. The inverse has an opposite sign.
    E = np.block([[1, np.zeros((1, active_size - 1))],
                  [(-a_21 / a_11)[:, np.newaxis], np.eye(active_size - 1)]])
    E_inv = np.block([[1, np.zeros((1, active_size - 1))],
                  [(a_21 / a_11)[:, np.newaxis], np.eye(active_size - 1)]])
    
    # Embed the elimination matrix and its inverse into the full size matrix
    E_full[offset:, offset:] = E
    E_inv_full[offset:, offset:] = E_inv
    # Update A
    A = E_full @ A

    # We need P_T, L, U from the recursive call
    p_rec, L, U = lu_decomposition(A, n, offset + 1)

    # Reconstruct the full permutation vector similarly to 5.iii.
    # Here the original matrix A ends up as U after applying the updates. 
    # L is the product of the inverse elimination matrices.
    return permute_indices(p_rec, indices), E_inv_full @ L, U

if __name__ == "__main__":
    # Test the LU decomposition with random matrices
    print(f"Testing LU decomposition with 8x200 random matrices...")
    for n in range(2, 10):
        for _ in range(100):
            A = np.random.rand(n, n)
            mx = np.sum(np.abs(A), axis=1)
            np.fill_diagonal(A, mx)

            A_cp = A.copy()
            p, L, U = lu_decomposition(A, n)
            A_cp[p] = A_cp[[x for x in range(n)]]
            # Check correctness numerically using np.allclose
            assert np.allclose(L @ U, A_cp), "LU decomposition failed"

            # # Check that the pivoting works using morally triangular matrices
            B = generate_random_morally_triangular_matrices(n)
            B_cp = B.copy()
            p, L, U = lu_decomposition(B, n)
            B_cp[p] = B_cp[[x for x in range(n)]]
            assert np.allclose(L @ U, B_cp), "LU decomposition failed"
    print("All tests passed!")