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


results = parallelizer.parallelize(normalize_rows)(
    np.random.rand(500_000, 25).astype(np.float32)
)
print(results)
