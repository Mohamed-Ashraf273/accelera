import time

import numpy as np
import psutil
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as skpipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from accelera.src.core.pipeline import Pipeline as accpipeline


def get_memory_info():
    """Get detailed memory information including swap"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_full = process.memory_full_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "swap_mb": (
            memory_full.swap / 1024 / 1024
            if hasattr(memory_full, "swap")
            else 0
        ),
    }


def sample_data():
    """Generate a big dataset for testing parallel execution performance."""
    print("Generating big dataset...")

    # Create a large, realistic dataset
    X, y = make_classification(
        n_samples=10_000,  # Large dataset
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


skpipe = skpipeline(
    [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
)


param_grid = [
    {
        "scaler": [StandardScaler(), MinMaxScaler()],
        "model": [LogisticRegression(max_iter=1000)],
    },
    {
        "scaler": [StandardScaler(), MinMaxScaler()],
        "model": [SVC(C=10)],
    },
]

# Total pipelines that will run:
# 1. StandardScaler -> LogisticRegression
# 2. MinMaxScaler -> LogisticRegression
# 3. StandardScaler -> SVC(C=10)
# 4. MinMaxScaler -> SVC(C=10)
# Total: 4 pipelines

acc_pipe = accpipeline()
acc_pipe.branch(
    "preprocessing",
    acc_pipe.preprocess("scaler", StandardScaler(), branch=True),
    acc_pipe.preprocess("scaler", MinMaxScaler(), branch=True),
).branch(
    "models",
    acc_pipe.model("model_lr", LogisticRegression(max_iter=1000), branch=True),
    acc_pipe.model("model_svc", SVC(C=10), branch=True),
).predict("predict", test_data=test_data).metric(
    "metric",
    "accuracy_score",
    y_true=y_test,
)

start_mem = get_memory_info()
start_time = time.time()
# search = GridSearchCV(
#     skpipe,
#     param_grid,
#     n_jobs=-1,
#     cv=2,
#     scoring='accuracy'  # Specify metric here
# )
# search.fit(X, y)
# print("\n=== Best Results ===")
# print(f"Best parameters: {search.best_params_}")
# print(f"Best cross-validation score: {search.best_score_:.4f}")
# print(f"Best estimator: {search.best_estimator_}")
predictions, best_path = acc_pipe(X, y, select_strategy="max")
predictions_f = best_path(test_data, y_test)
print(predictions_f)
end_time = time.time()
end_mem = get_memory_info()
print("Memory usage")
print(f"RSS memory used: {end_mem['rss_mb'] - start_mem['rss_mb']:.2f} MB")
print(f"VMS memory used: {end_mem['vms_mb'] - start_mem['vms_mb']:.2f} MB")
print(f"Swap memory used: {end_mem['swap_mb'] - start_mem['swap_mb']:.2f} MB")
print(f"Time taken: {end_time - start_time:.2f} seconds")
