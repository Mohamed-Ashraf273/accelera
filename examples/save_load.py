import pickle
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from accelera.src.accelera_pipe.core.pipeline import Pipeline

# Demo 4
print("\n" * 2 + "=" * 80)
print("===Save and Load Executed Pipeline Demo===")
print(
    "File running: examples/save_load.py\n"
    "Protocol: Generate a large synthetic dataset, \n"
    "define a pipeline with preprocessing and model training, \n"
    "execute it, save the executed pipeline, load it back, \n"
    "and evaluate on test data."
)
print("=" * 80)


def sample_data(random_state=42, n_samples=50_000):
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


X, y, X_val, y_val = sample_data()
X_test, y_test, _, _ = sample_data(
    random_state=24, n_samples=1000
)  # Different random state for test data


def test_save_load_executed_pipe():
    # Sklearn
    start_sk_time = time.time()
    p1 = StandardScaler()
    p2 = MinMaxScaler()
    p1.fit(X)
    p2.fit(X)
    X1 = p1.transform(X)
    X2 = p2.transform(X)

    model1 = LogisticRegression(max_iter=1000)
    model1.fit(X1, y)
    model2 = LogisticRegression(max_iter=1000)
    model2.fit(X2, y)

    val1 = p1.transform(X_val)
    val2 = p2.transform(X_val)
    pred1 = model1.predict(val1)
    pred2 = model2.predict(val2)

    if accuracy_score(y_val, pred1) > accuracy_score(y_val, pred2):
        pickle.dump((p1, model1), open("best_model.pkl", "wb"))
    else:
        pickle.dump((p2, model2), open("best_model.pkl", "wb"))

    final_pipe = pickle.load(open("best_model.pkl", "rb"))

    final_test = final_pipe[0].transform(X_test)
    final_pred = final_pipe[1].predict(final_test)
    print("==============Sklearn Results Of Loaded Pipeline==============")
    print(accuracy_score(y_test, final_pred))
    print(
        f"Sklearn pipeline execution time: {time.time() - start_sk_time:.2f} seconds"
    )

    # Accelera
    start_acc_time = time.time()
    accpipe = Pipeline()
    accpipe.branch(
        "preprocessing",
        accpipe.preprocess("scaler", MinMaxScaler(), branch=True),
        accpipe.preprocess("scaler", StandardScaler(), branch=True),
    ).model("model_lr", LogisticRegression(max_iter=1000)).predict(
        "predict", test_data=X_val
    ).metric(
        "metric",
        "accuracy_score",
        y_true=y_val,
    )
    accelerated_results, executed_graph = accpipe(X, y, select_strategy="max")
    executed_graph.save()

    with open("pipeline.pkl", "rb") as f:
        loaded_executed_graph = pickle.load(f)

    print("==============Accelera Results Of Loaded Executed Pipeline==============")
    print(loaded_executed_graph(X_test, y_true=y_test)[0]["result"])
    elapsed = time.time() - start_acc_time
    print(f"Accelera pipeline execution time: {elapsed:.2f} seconds")


def test_save_load_pipeline():
    accpipe = Pipeline()
    accpipe.branch(
        "preprocessing",
        accpipe.preprocess("scaler", MinMaxScaler(), branch=True),
        accpipe.preprocess("scaler", StandardScaler(), branch=True),
    ).model("model_lr", LogisticRegression(max_iter=1000)).predict(
        "predict", test_data=X_val
    ).metric(
        "metric",
        "accuracy_score",
        y_true=y_val,
    )
    accpipe.save("pipeline.pkl")

    with open("pipeline.pkl", "rb") as f:
        loaded_pipeline = pickle.load(f)

    start_time = time.time()
    _, loaded_executed_graph = loaded_pipeline(X, y, select_strategy="max")
    print("==============Accelera Results Of Loaded Pipeline==============")
    print(loaded_executed_graph(X_test, y_true=y_test)[0]["result"])
    elapsed = time.time() - start_time
    print(f"Accelera pipeline execution time: {elapsed:.2f} seconds")


test_save_load_executed_pipe()
test_save_load_pipeline()
