import time

import numpy as np

from accelera.src.utils.parallelizer import parallelizer


def normalize_rows(X):
    for i in range(len(X)):
        s = 0
        for j in range(len(X[i])):
            s += X[i][j] * X[i][j]

        norm = s**0.5

        for j in range(len(X[i])):
            X[i][j] = X[i][j] / norm

    return X


X = np.random.rand(500_000, 25).astype(np.float32)

# Accelera parallelized version
start_time = time.time()
acc_results = parallelizer.parallelize(normalize_rows)(X)
end_time = time.time()
print(f"Execution time Accelera: {end_time - start_time} seconds")


# Non-Accelera version for comparison
start_time = time.time()
results = normalize_rows(X)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")


# Correctness check
if np.allclose(results, acc_results):
    print("Validation successful: Accelera results match non-Accelera results.")
else:
    print("Validation failed: Accelera results do not match non-Accelera results.")
