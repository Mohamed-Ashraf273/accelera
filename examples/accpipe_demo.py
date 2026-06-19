import time

import numpy as np
import psutil
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from accelera.src.accelera_pipe.core.pipeline import Pipeline as accpipeline
from accelera.src.accelera_pipe.wrappers.graph_report import GraphReport
from accelera.src.utils.accelera_utils import serialize

# Demo 1
print("=" * 80)
print("===Accelera Pipe Demo: Comparing against sklearn pipelines===")
print(
    "File running: examples/accpipe_demo.py\n"
    "Protocol: same train/val/test split, same 4 candidate pipelines, \n"
    "selection on val accuracy, report on test accuracy"
)
print("=" * 80)


def get_memory_info():
    """Get detailed memory information including swap"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_full = process.memory_full_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "swap_mb": (
            memory_full.swap / 1024 / 1024 if hasattr(memory_full, "swap") else 0
        ),
    }


def sample_data():
    """Generate a big dataset for testing parallel execution performance."""
    print("Generating big dataset...")

    # Create a large, realistic dataset
    X, y = make_classification(
        n_samples=25_000,  # Large dataset
        n_features=25,  # High-dimensional features
        n_classes=4,  # Multi-class problem
        n_informative=20,  # Most features are informative
        n_redundant=3,  # Some redundant features
        n_clusters_per_class=2,  # Complex class structure
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    print(
        f"Dataset created: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}"
    )
    print(f"Feature range: {X.min():.3f} to {X.max():.3f}")
    print(f"Classes: {np.unique(y)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = sample_data()


candidates = [
    (StandardScaler(), LogisticRegression(max_iter=1000)),
    (MinMaxScaler(), LogisticRegression(max_iter=1000)),
    (StandardScaler(), SVC(C=10)),
    (MinMaxScaler(), SVC(C=10)),
]

# Total pipelines that will run:
# 1. StandardScaler -> LogisticRegression
# 2. MinMaxScaler -> LogisticRegression
# 3. StandardScaler -> SVC(C=10)
# 4. MinMaxScaler -> SVC(C=10)
# Total: 4 pipelines


def extract_metric(metric_result):
    if isinstance(metric_result, dict) and "result" in metric_result:
        return float(metric_result["result"])
    return float(metric_result)


sk_start_mem = get_memory_info()
sk_start_time = time.time()
best_sk = None

for scaler, model in candidates:
    scaler_inst = clone(scaler)
    model_inst = clone(model)

    X_train_scaled = scaler_inst.fit_transform(X_train)
    X_val_scaled = scaler_inst.transform(X_val)

    model_inst.fit(X_train_scaled, y_train)
    y_val_pred = model_inst.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)

    if best_sk is None or val_acc > best_sk["val_acc"]:
        best_sk = {
            "scaler": scaler_inst.__class__.__name__,
            "model": model_inst.__class__.__name__,
            "scaler_inst": scaler_inst,
            "model_inst": model_inst,
            "val_acc": val_acc,
        }

X_test_scaled = best_sk["scaler_inst"].transform(X_test)
y_test_pred = best_sk["model_inst"].predict(X_test_scaled)

best_sk["test_acc"] = accuracy_score(y_test, y_test_pred)
sk_end_time = time.time()
sk_end_mem = get_memory_info()

sk_pipe_start_mem = get_memory_info()
sk_pipe_start_time = time.time()
best_sk_pipe = None

for scaler, model in candidates:
    pipe = SklearnPipeline(
        [
            ("scaler", clone(scaler)),
            ("model", clone(model)),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    if best_sk_pipe is None or val_acc > best_sk_pipe["val_acc"]:
        best_sk_pipe = {
            "scaler": pipe.named_steps["scaler"].__class__.__name__,
            "model": pipe.named_steps["model"].__class__.__name__,
            "pipe": pipe,
            "val_acc": val_acc,
        }

y_test_pred = best_sk_pipe["pipe"].predict(X_test)
best_sk_pipe["test_acc"] = accuracy_score(y_test, y_test_pred)
sk_pipe_end_time = time.time()
sk_pipe_end_mem = get_memory_info()

acc_start_mem = get_memory_info()
acc_start_time = time.time()
acc_pipe = accpipeline()
acc_pipe.branch(
    "preprocessing",
    acc_pipe.preprocess("scaler", StandardScaler(), branch=True),
    acc_pipe.preprocess("scaler", MinMaxScaler(), branch=True),
).branch(
    "models",
    acc_pipe.model("model_lr", LogisticRegression(max_iter=1000), branch=True),
    acc_pipe.model("model_svc", SVC(C=10), branch=True),
).predict("predict", test_data=X_val).metric(
    "metric",
    "accuracy_score",
    y_true=y_val,
)

acc_val_results, best_path = acc_pipe(X_train, y_train, select_strategy="max")
acc_test_results = best_path(X_test, y_true=y_test)
acc_end_time = time.time()
acc_end_mem = get_memory_info()

serialize(acc_pipe, "accelera_pipe.xml")
report = GraphReport("report", "accelera_pipe.xml", acc_val_results)
img_path = report.execute()

acc_val_acc = max(extract_metric(result) for result in acc_val_results)
acc_test_acc = extract_metric(acc_test_results[0])
sk_time = sk_end_time - sk_start_time
sk_pipe_time = sk_pipe_end_time - sk_pipe_start_time
acc_time = acc_end_time - acc_start_time

print("\n===Comparison Report ===")
print(
    "Protocol: same train/val/test split, "
    "same 4 candidate pipelines, selection on val "
    "accuracy, report on test accuracy"
)
print("\n[sklearn]")
print(
    f"Best candidate: {best_sk['scaler']} -> {best_sk['model']} | "
    f"val_acc={best_sk['val_acc']:.4f} | test_acc={best_sk['test_acc']:.4f}"
)
print(f"Time: {sk_time:.2f} s")
print(
    f"Memory delta (RSS/VMS/Swap MB): "
    f"{sk_end_mem['rss_mb'] - sk_start_mem['rss_mb']:.2f} / "
    f"{sk_end_mem['vms_mb'] - sk_start_mem['vms_mb']:.2f} / "
    f"{sk_end_mem['swap_mb'] - sk_start_mem['swap_mb']:.2f}"
)

print("\n[sklearn Pipeline]")
print(
    f"Best candidate: {best_sk_pipe['scaler']} -> {best_sk_pipe['model']} | "
    f"val_acc={best_sk_pipe['val_acc']:.4f} | "
    f"test_acc={best_sk_pipe['test_acc']:.4f}"
)
print(f"Time: {sk_pipe_time:.2f} s")
print(
    f"Memory delta (RSS/VMS/Swap MB): "
    f"{sk_pipe_end_mem['rss_mb'] - sk_pipe_start_mem['rss_mb']:.2f} / "
    f"{sk_pipe_end_mem['vms_mb'] - sk_pipe_start_mem['vms_mb']:.2f} / "
    f"{sk_pipe_end_mem['swap_mb'] - sk_pipe_start_mem['swap_mb']:.2f}"
)

print("\n[accelera]")
print(f"Selected path val_acc={acc_val_acc:.4f} | test_acc={acc_test_acc:.4f}")
print(f"Time: {acc_time:.2f} s")
print(
    f"Memory delta (RSS/VMS/Swap MB): "
    f"{acc_end_mem['rss_mb'] - acc_start_mem['rss_mb']:.2f} / "
    f"{acc_end_mem['vms_mb'] - acc_start_mem['vms_mb']:.2f} / "
    f"{acc_end_mem['swap_mb'] - acc_start_mem['swap_mb']:.2f}"
)

print("\n[summary]")
print(
    f"Test accuracy delta (accelera - sklearn): "
    f"{acc_test_acc - best_sk['test_acc']:+.4f}"
)
print(f"Time delta (accelera - sklearn): {acc_time - sk_time:+.2f} s")
print(f"Time delta (accelera - sklearn Pipeline): {acc_time - sk_pipe_time:+.2f} s")
