import numpy as np
from numpy.typing import NDArray
from exercise_5.generate_random_morally_triangular_matrices_5_i import generate_random_morally_triangular_matrices

def lu_decomposition(A: NDArray, n: int, offset: int = 0):
    P_T = np.eye(n)
    if offset == n - 1: return P_T, np.eye(n), A # Base case
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
    
        # Construct the permutation matrix
        P_T = np.eye(n)
        P_T[offset, offset] = 0
        P_T[smallest_row_index, smallest_row_index] = 0
        P_T[offset, smallest_row_index] = 1
        P_T[smallest_row_index, offset] = 1

    A = P_T @ A  # Apply the permutation to A

    # Get the vectors for the elimination matrix
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
    P_T_rec, L, U = lu_decomposition(A, n, offset + 1)

    # Update P_T to get the full permutation matrix. 
    # Here the original matrix A ends up as U after applying the updates. 
    # L is the product of the inverse elimination matrices.
    return P_T_rec @ P_T, E_inv_full @ L, U

if __name__ == "__main__":
    # Test the LU decomposition with random matrices
    print(f"Testing LU decomposition with 8x200 random matrices...")
    for n in range(3, 10):
        for _ in range(100):
            # Generate random invertible matrix
            A = np.random.rand(n, n)
            mx = np.sum(np.abs(A), axis=1)
            np.fill_diagonal(A, mx)

            P, L, U = lu_decomposition(A, n)
            # Check correctness numerically using np.allclose
            assert np.allclose(P.T @ L @ U, A), "LU decomposition failed"

            # Check that the pivoting works using morally triangular matrices
            B = generate_random_morally_triangular_matrices(n)
            P, L, U = lu_decomposition(B, n)

            assert np.allclose(P.T @ L @ U, B), "LU decomposition failed"
    print("All tests passed!")