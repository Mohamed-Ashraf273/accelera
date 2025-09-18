import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import pickle, hashlib, os


# ---------- Cache Manager ----------
class CacheManager:
    def __init__(self, cache_dir="cache"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def _get_hash(self, df: pd.DataFrame) -> str:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def load(self, name: str, df: pd.DataFrame):
        h = self._get_hash(df)
        cache_file = os.path.join(self.cache_dir, f"{name}_{h}.pkl")
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, "rb"))
        return None

    def save(self, name: str, df: pd.DataFrame, params):
        h = self._get_hash(df)
        cache_file = os.path.join(self.cache_dir, f"{name}_{h}.pkl")
        pickle.dump(params, open(cache_file, "wb"))


# ---------- Cached PCA ----------
class CachedPCA:
    def __init__(self, n_components, cache_manager: CacheManager):
        self.n_components = n_components
        self.cache_manager = cache_manager
        self.params = None

    def fit(self, df: pd.DataFrame):
        cached = self.cache_manager.load("pca", df)
        if cached:
            print("✅ Using cached PCA")
            self.params = cached
        else:
            print("⚙️  Computing PCA...")
            pca = PCA(n_components=self.n_components)
            pca.fit(df)
            self.params = {"mean": pca.mean_, "components": pca.components_}
            self.cache_manager.save("pca", df, self.params)
        return self

    def transform(self, df: pd.DataFrame):
        if not self.params:
            raise ValueError("PCA not fitted yet!")
        return (df - self.params["mean"]) @ self.params["components"].T


# ---------- Test with Big Data ----------
# create large dataset (100k samples, 200 features)
X, _ = make_classification(n_samples=100_000, n_features=200, random_state=42)
df = pd.DataFrame(X)

# Without cache
t1 = time.time()
for _ in range(2):
    pca = PCA(n_components=50)
    pca.fit(df)
    pca.transform(df)
t2 = time.time()

# With cache
cache = CacheManager()
scaler_pca = CachedPCA(n_components=50, cache_manager=cache)
t3 = time.time()
for _ in range(2):
    scaler_pca.fit(df)
    scaler_pca.transform(df)
t4 = time.time()

print("⏱ No Cache Time:", round(t2 - t1, 2), "seconds")
print("⚡ With Cache Time:", round(t4 - t3, 2), "seconds")
