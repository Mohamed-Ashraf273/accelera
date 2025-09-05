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
    test_data = X[:500]  # Use first 500 samples for testing

    print(
        f"Dataset created: {X.shape} "
        f"training samples, {test_data.shape} test samples"
    )
    print(f"Feature range: {X.min():.3f} to {X.max():.3f}")
    print(f"Classes: {np.unique(y)}")

    return X, y, test_data


X, y, test_data = sample_data()

p = Pipeline()
# p._graph.enableParallelExecution(True) # not working yet


print("\nSetting up preprocessing branches...")
scaler = StandardScaler()
scaler.fit(X)  # Pre-fit the scaler

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

p.preprocess("feature_engineering", lambda x: np.clip(x, -5, 5))

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
# p.merge("ensemble_average", lambda preds: np.mean(preds, axis=0)
# if isinstance(preds, list) and len(preds) > 1 else
# (preds[0] if isinstance(preds, list) else preds))

p.serialize("big_data_pipeline_no_merge.xml")

print("\nExecuting big data pipeline...")
start = time.time()
predictions = p(X, y)
end = time.time()

print("\nResults:")
print(
    "Predictions shape: "
    f"{
        predictions[0].shape
        if predictions and hasattr(predictions[0], 'shape')
        else 'Unknown'
    }"
)
print(
    "Sample predictions: "
    f"{
        predictions[0][:10]
        if predictions and hasattr(predictions[0], '__getitem__')
        else predictions
    }"
)
print(f"Execution time: {end - start:.3f} seconds")
print(f"Throughput: {len(X) / (end - start):.0f} samples/second")

# Performance analysis
if end - start > 0:
    print("\nPerformance Analysis:")
    print(f"- Dataset size: {X.shape}")
    print("- Models trained: 3 (LogisticRegression, SVM, RandomForest)")
    print(
        "- Preprocessing steps: 4 (scaling, "
        "normalization, power transform, clipping)"
    )
    print(f"- Parallel execution: {'Enabled' if p._graph else 'Disabled'}")
    print(f"- Total time: {end - start:.3f}s")

    if end - start < 10:
        print("🚀 Excellent performance!")
    elif end - start < 30:
        print("✅ Good performance")
    else:
        print("⚠️  Consider optimization for this dataset size")
