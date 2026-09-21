import numpy as np
from week_3.exercise_4.ex_4_i import base_b
from week_3.exercise_4.ex_4_ii import generate_base_b

if __name__ == "__main__":
    # Run 100 tests with different bases and precisions
    for _ in range(100):
        for b in range(2, 10):
            for N in range(2, 10):
                    base_ten, base_b_ = generate_base_b(b, N)
                    base_b_repr = base_b(b, base_ten, N)

                    assert np.allclose(base_b_, base_b_repr), f"b: {b}, base_b_: {base_b_}, base_b_repr: {base_b_repr}"

    print("All tests passed!")