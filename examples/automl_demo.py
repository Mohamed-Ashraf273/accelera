import numpy as np
from sklearn.datasets import make_classification

from mainera.src.automl import AutoMLAgent
from mainera.src.automl.utils.sampling import sample


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

# Simple usage
agent = AutoMLAgent()
# agent.fit(X, y, task="classification")
# best_pipeline, report = agent.fit(X, y, task="classification")
# predictions = best_pipeline.predict(test_data)

# # Advanced usage with constraints
# agent = AutoMLAgent(
#     time_limit=300,  # 5 minutes
#     metric="f1_score",
#     n_trials=50,
#     use_llm=True,  # Use your LLM models for suggestions
# )
# results = agent.fit(X, y, task="classification")

# test sampling
X_sampled, y_sampled, meta_data = sample(X, y)
print(len(X_sampled))
print(meta_data["original_distribution"])
print(meta_data["final_distribution"])
