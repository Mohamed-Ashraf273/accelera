import os
import time
import pickle
import hashlib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


class PickleModelCache:
    def __init__(self, cache_dir="cache/model_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _make_key(self, X, y, params):
        data_hash = hashlib.sha256(pickle.dumps((X, y, params))).hexdigest()
        return data_hash

    def get_or_train(self, X, y, params):
        key = self._make_key(X, y, params)
        file_path = os.path.join(self.cache_dir, f"{key}.pkl")

        if os.path.exists(file_path):
            print("🔄 Loading model from cache...")
            with open(file_path, "rb") as f:
                model = pickle.load(f)
            return model

        print("⚡ Training new model...")
        model = LogisticRegression(**params, max_iter=2000)
        start = time.time()
        model.fit(X, y)
        elapsed = time.time() - start
        print(f"Training took {elapsed:.3f} seconds")

        # نخزن الموديل
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        return model

X, y = make_classification(n_samples=50000, n_features=100, random_state=42)

cache = PickleModelCache()

params = {"solver": "lbfgs"}

model1 = cache.get_or_train(X, y, params)

model2 = cache.get_or_train(X, y, params)

params2 = {"solver": "liblinear"}
model3 = cache.get_or_train(X, y, params2)
