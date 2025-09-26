import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mainera.src.core.pipeline import Pipeline
from mainera.src.custom.classifier import CustomClassifier


class TorchDenseModel(CustomClassifier):
    def __init__(
        self,
        input_dim=None,
        hidden_dim=32,
        lr=0.01,
        epochs=100,
        random_state=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

        self.model = None
        print("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _build_model(self, input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes),
        ).to(self.device)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        if self.model is None:
            self.model = self._build_model(n_features, num_classes).to(
                self.device
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, device=self.device)
        y_tensor = torch.tensor(y, device=self.device)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        return preds

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        return probabilities


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
    y_test = y[:500]
    print(
        f"Dataset created: {X.shape} "
        f"training samples, {test_data.shape} test samples"
    )
    print(f"Feature range: {X.min():.3f} to {X.max():.3f}")
    print(f"Classes: {np.unique(y)}")

    return X, y, test_data, y_test


X, y, test_data, y_test = sample_data()


print("\nSetting up preprocessing branches...")
scaler = StandardScaler()
scaler.fit(X)  # Pre-fit the scaler

p1 = scaler.transform


def p_common(x):
    return np.clip(x, -5, 5)


def p2(x):
    return np.sign(x) * np.power(np.abs(x), 0.8)


m1 = LogisticRegression(random_state=42, max_iter=1000)
m2 = SVC(probability=True, random_state=42, kernel="rbf")
m3 = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)


pre_processed = p_common(p2(p1(X)))
m1.fit(pre_processed, y)

p = Pipeline()

# p.disable_parallel_execution()

p.branch(
    "preprocessing",
    [
        p.preprocess("standard_scaler", StandardScaler(), branch=True),
        p.preprocess(
            "power_transform",
            lambda x: np.sign(x) * np.power(np.abs(x), 0.8),
            branch=True,
        ),
    ],
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
    p.model("custom", TorchDenseModel(random_state=42), branch=True),
    p.model(
        "rf",
        RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
        branch=True,
    ),
)

p.predict("predict", test_data)
p.metric("classification_report", "classification_report", y_test)
p.serialize("test.xml")
start_mem = get_memory_info()
start = time.time()
simple_predictions, executed_graph = p(X, y)
end = time.time()
end_mem = get_memory_info()

print(f"Pipeline execution time: {end - start:.4f} seconds")
print(f"RSS memory: {end_mem['rss_mb'] - start_mem['rss_mb']:.2f} MB increase")
print(f"Swap memory used: {end_mem['swap_mb']:.2f} MB")
print(executed_graph(test_data, metrics=True, y_true=y_test)[0])
print("Sample predictions: ", executed_graph(test_data)[0])
print(f"{simple_predictions[0]}")
