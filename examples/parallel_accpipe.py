import time

import numpy as np
from sklearn.datasets import make_classification

from accelera.src.accelera_pipe.core.pipeline import Pipeline


def sample_data(random_state=42, n_samples=500_000):
    """Generate a big dataset for testing parallel execution performance."""
    print("Generating big dataset...")

    # Create a large, realistic dataset
    X, y = make_classification(
        n_samples=n_samples,  # Large dataset
        n_features=25,  # High-dimensional features
        n_classes=4,  # Multi-class problem
        n_informative=20,  # Most features are informative
        n_redundant=3,  # Some redundant features
        n_clusters_per_class=2,  # Complex class structure
        random_state=random_state,
    )

    # Create test data (subset of training data)
    test_data = X[:500]  # Use first 50 samples for testing
    y_test = y[:500]
    print(
        f"Dataset created: {X.shape} "
        f"training samples, {test_data.shape} test samples"
    )
    print(f"Feature range: {X.min():.3f} to {X.max():.3f}")
    print(f"Classes: {np.unique(y)}")

    return X, y, test_data, y_test


def normalize_rows(X):
    for i in range(len(X)):
        s = 0
        for j in range(len(X[i])):
            s += X[i][j] * X[i][j]

        norm = s**0.5

        for j in range(len(X[i])):
            X[i][j] = X[i][j] / norm

    return X


X, y, X_val, y_val = sample_data()
start_time = time.time()
accpipe = Pipeline()
accpipe.preprocess("normalize", normalize_rows)
res, executed_graph = accpipe(X, y, select_strategy="max")
print("==============Accelera Results==============")
elapsed = time.time() - start_time
print(f"Accelera pipeline execution time: {elapsed:.2f} seconds")
print(res[0])
