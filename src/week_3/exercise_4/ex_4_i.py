import numpy as np
from numpy.typing import NDArray

def base_b(b: int, x: float, N: int, base_b_repr: NDArray = np.array([])):
    machine_epsilon = 1 / (b ** (N - 1))
    assert b in range(2, 10), "b must be between 2 and 9"
    assert x + machine_epsilon >= 0 and x < b + machine_epsilon, f"{x} must be betwen 0 and b"

    if N == -1: return base_b_repr # Return the array of N digits. We return at -1 since generate_base_b generates N+1 digits.

    # Python native int() methods is the same as np.floor. We round to N digits to avoid floating point errors.
    base_b_repr = np.append(base_b_repr, int(round(x, N)))

    # Recurse with b(x-floor(x)) and N-1 digits. Also, we pass the array as a parameter
    return base_b(b, b * (x - int(round(x, N))), N - 1, base_b_repr)

if __name__ == "__main__":
    base_b_repr = base_b(2, 1.101001, 6)
    print(base_b_repr)