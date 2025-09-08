import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mainera.src.core.pipeline import Pipeline


def sample_data():
    """Generate a big dataset for testing parallel execution performance."""
    print("Generating big dataset...")

    # Create a large, realistic dataset
    X, y = make_classification(
        n_samples=10000,  # Large dataset
        n_features=25,  # High-dimensional features
        n_classes=4,  # Multi-class problem
        n_informative=20,  # Most features are informative
        n_redundant=3,  # Some redundant features
        n_clusters_per_class=2,  # Complex class structure
        random_state=42,
    )

    # Create test data (subset of training data)
    test_data = X[:500]  # Use first 50 samples for testing

    print(
        f"Dataset created: {X.shape} "
        f"training samples, {test_data.shape} test samples"
    )
    print(f"Feature range: {X.min():.3f} to {X.max():.3f}")
    print(f"Classes: {np.unique(y)}")

    return X, y, test_data


X, y, test_data = sample_data()


print("\nSetting up preprocessing branches...")
scaler = StandardScaler()
scaler.fit(X)  # Pre-fit the scaler

p1 = scaler.transform


def p_common(x):
    return np.clip(x, -5, 5)


m1 = LogisticRegression(random_state=42, max_iter=1000)
m2 = SVC(probability=True, random_state=42, kernel="rbf")
m3 = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
pre_processed = p_common(p1(X))
m3.fit(pre_processed, y)

p = Pipeline()

# p.disable_parallel_execution()

p.branch(
    "preprocessing",
    p.preprocess("standard_scaler", scaler.transform, branch=True),
    p.preprocess(
        "normalize",
        lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8),
        branch=True,
    ),
    p.preprocess(
        "power_transform",
        lambda x: np.sign(x) * np.power(np.abs(x), 0.8),
        branch=True,
    ),
)
p.preprocess("clip", lambda x: np.clip(x, -5, 5))

p.branch(
    "models",
    p.model(
        "logreg",
        LogisticRegression(random_state=42, max_iter=1000),
        branch=True,
    ),
    p.model(
        "svm", SVC(probability=True, random_state=42, kernel="rbf"), branch=True
    ),
    p.model(
        "rf",
        RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
        branch=True,
    ),
)

p.predict("predict", test_data)
p.serialize("test.xml")
start = time.time()
simple_predictions = p(X, y)
end = time.time()
print(f"Pipeline execution time: {end - start:.4f} seconds")

print(
    "Pipeline matches: "
    f"{np.all(m3.predict(p_common(p1(test_data))) == simple_predictions[2])}"
)
