import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from mainera.src.automl.utils.sampling import sample
from mainera.src.core.pipeline import Pipeline

p1 = Pipeline()
p2 = Pipeline()


def sample_data():
    """Generate a big dataset for testing parallel execution performance."""
    print("Generating big dataset...")

    # Create a large, realistic dataset
    X, y = make_classification(
        n_samples=100000,  # Large dataset
        n_features=25,  # High-dimensional features
        n_classes=4,  # Multi-class problem
        n_informative=20,  # Most features are informative
        n_redundant=3,  # Some redundant features
        n_clusters_per_class=2,  # Complex class structure
        random_state=42,
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


X, y, test_data, y_test = sample_data()


X_sampled, y_sampled, meta_data = sample(X, y)
p1.preprocess("p1_scaling", StandardScaler()).model(
    "rf1", RandomForestClassifier(n_estimators=10, random_state=42)
).predict("predict", test_data=test_data).metric(
    "report", "classification_report", y_true=y_test
)
output_1, _ = p1(X, y)
p2.preprocess("p2_scaling", StandardScaler()).model(
    "rf2", RandomForestClassifier(n_estimators=10, random_state=42)
).predict("predict", test_data=test_data).metric(
    "report", "classification_report", y_true=y_test
)
output_2, _ = p2(X_sampled, y_sampled)
print(output_1[0]["result"])
print(meta_data["original_size"])
print(output_2[0]["result"])
print(meta_data["final_size"])
