import numpy as np

def generate_s_p_s_d(n: int, k: int):

    # Generate a random n x n matrix
    F = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            F[i, j] = np.random.rand()

    Q, _ = np.linalg.qr(F)

    # Same as the Matlab code
    P = Q[:, :(n - k)] @ Q[:, :(n - k)].T

    # Using the same strategy to generate s.p.d matrices as in 6. iii).
    F = np.zeros((n, n))
        
    for i in range(n):
        for j in range(n):
            F[i, j] = np.random.rand()

    A = F.T @ F

    # Using 3. ii)
    rand_s_p_s_d = P.T @ A @ P

    assert np.allclose(rand_s_p_s_d, rand_s_p_s_d.T) # Check symmetricity
    assert all(i + 1e-09 >= 0 for i in np.linalg.eigvals(rand_s_p_s_d)) # Check positive semidefiniteness
    assert np.linalg.matrix_rank(rand_s_p_s_d) == n - k # Check that rank is n - k

    return rand_s_p_s_d

if __name__ == "__main__":
    rand_s_p_s_d = generate_s_p_s_d(5, 2)
    print(rand_s_p_s_d)