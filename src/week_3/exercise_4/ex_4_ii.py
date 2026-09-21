import numpy as np
from functools import reduce

def generate_base_b(b: int, N: int):
    base_b = np.floor((b * np.random.rand(N + 1)))
    exponents = range(N + 1)

    # This is a shorthand for writing \sum_{i = 0}^N base^{-i} * c_i
    base_ten = reduce(lambda x, y: x + b ** (-y[1]) * y[0] , zip(base_b, exponents), 0)

    return base_ten, base_b

if __name__ == "__main__":
    base_ten, base_b = generate_base_b(2, 6)
    print(base_ten)
    print(base_b)