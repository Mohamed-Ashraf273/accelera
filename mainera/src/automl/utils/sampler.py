import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from mainera.src.utils.array_utils import convert_to_array
from mainera.src.utils.mainera_utils import print_msg


class Sampler:
    DEFAULT_SAMPLING_RATIO = 0.5
    MIN_SAMPLES_PER_CLASS = 500

    def __init__(
        self,
        target_size=None,
        sampling_ratio=None,
        min_samples_per_class=None,
        random_state=42,
    ):
        self.target_size = target_size
        self.sampling_ratio = sampling_ratio or self.DEFAULT_SAMPLING_RATIO
        self.min_samples_per_class = (
            min_samples_per_class or self.MIN_SAMPLES_PER_CLASS
        )
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def sample(self, X, y=None):
        """
        Sample data while preserving important patterns.

        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target vector (optional for classification)

        Returns:
            tuple: (X_sampled, y_sampled, metadata)
        """
        X_array = convert_to_array(X)
        y_array = convert_to_array(y) if y is not None else None
        n_samples = len(X_array)

        if self.target_size is not None:
            target = min(self.target_size, n_samples)
        else:
            target = int(n_samples * self.sampling_ratio)

        if target >= n_samples * 0.95:
            return (
                X,
                y,
                self._create_metadata(
                    n_samples, n_samples, y_array, "no_sampling"
                ),
            )

        if y_array is not None and self._is_classification(y_array):
            indices = self._stratified_sample(X_array, y_array, target)
            strategy = "stratified_clustering"
        else:
            indices = self._random_sample(X_array, target)
            strategy = "random_sampling"

        X_sampled = self._extract(X, indices)
        y_sampled = self._extract(y, indices) if y is not None else None

        return (
            X_sampled,
            y_sampled,
            self._create_metadata(
                n_samples, len(indices), y_array, strategy, y_sampled
            ),
        )

    def _stratified_sample(self, X, y, target_size):
        unique_classes, class_counts = np.unique(y, return_counts=True)

        samples_per_class = {}
        for cls, count in zip(unique_classes, class_counts):
            proportional = int((count / len(X)) * target_size)
            samples_per_class[cls] = max(
                proportional, min(self.min_samples_per_class, count)
            )

        total = sum(samples_per_class.values())
        err = abs(total - target_size)
        max_err = target_size * 0.1
        max_iterations = 20
        iterations = 0
        while err > max_err and iterations < max_iterations:
            scale = target_size / total
            samples_per_class = {
                cls: max(
                    min(self.min_samples_per_class, class_counts[i]),
                    round(n * scale),
                )
                for i, (cls, n) in enumerate(samples_per_class.items())
            }
            total = sum(samples_per_class.values())
            err = abs(total - target_size)
            iterations += 1

        all_indices = []
        for cls in unique_classes:
            class_mask = y == cls
            class_indices = np.where(class_mask)[0]
            n_needed = min(samples_per_class[cls], len(class_indices))

            if len(class_indices) <= n_needed:
                all_indices.extend(class_indices)
            else:
                selected = self._sample_class(
                    X[class_mask], class_indices, n_needed
                )
                all_indices.extend(selected)

        return np.array(all_indices)

    def _sample_class(self, X_class, class_indices, n_samples):
        """
        Advanced sampling from a single class.

        Strategy:
        - 50% cluster centers (representative core samples)
        - 25% boundary samples (outliers/hard cases)
        - 25% random samples (diversity)
        """
        n_clusters_samples = int(n_samples * 0.5)
        n_boundary = int(n_samples * 0.25)
        n_random = n_samples - n_clusters_samples - n_boundary

        n_clusters = min(
            max(n_clusters_samples // 2, 5), len(X_class) // 5, 100
        )

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_class)

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=min(1000, len(X_class)),
                n_init=5,
                max_iter=100,
            )
            labels = kmeans.fit_predict(X_scaled)

            # 1. Select cluster center samples (50%)
            cluster_samples = []
            samples_per_cluster = max(1, n_clusters_samples // n_clusters)

            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                cluster_size = mask.sum()

                if cluster_size == 0:
                    continue

                cluster_idx = np.where(mask)[0]
                center = kmeans.cluster_centers_[cluster_id]

                distances = np.linalg.norm(
                    X_scaled[cluster_idx] - center, axis=1
                )
                n_take = min(samples_per_cluster, cluster_size)
                closest = cluster_idx[np.argsort(distances)[:n_take]]
                cluster_samples.extend(class_indices[closest])

            # 2. Select boundary samples (25%) - outliers and edge cases
            boundary_samples = []
            if n_boundary > 0:
                all_centers = kmeans.cluster_centers_[labels]
                distances_to_center = np.linalg.norm(
                    X_scaled - all_centers, axis=1
                )

                available_mask = np.ones(len(X_class), dtype=bool)
                for idx in cluster_samples:
                    local_idx = np.where(class_indices == idx)[0]
                    if len(local_idx) > 0:
                        available_mask[local_idx[0]] = False

                available_indices = np.where(available_mask)[0]
                if len(available_indices) > 0:
                    boundary_distances = distances_to_center[available_indices]
                    n_boundary_take = min(n_boundary, len(available_indices))
                    boundary_idx = available_indices[
                        np.argsort(boundary_distances)[-n_boundary_take:]
                    ]
                    boundary_samples.extend(class_indices[boundary_idx])

            # 3. Add random samples (25%) for diversity
            random_samples = []
            if n_random > 0:
                selected_set = set(cluster_samples + boundary_samples)
                available = [
                    idx for idx in class_indices if idx not in selected_set
                ]

                if available:
                    n_random_take = min(n_random, len(available))
                    random_idx = self.rng.choice(
                        available, size=n_random_take, replace=False
                    )
                    random_samples.extend(random_idx)

            all_selected = cluster_samples + boundary_samples + random_samples
            return all_selected[:n_samples]

        except Exception as e:
            print_msg(
                f"Advanced sampling failed, using random: {e}", level="warning"
            )
            return self._random_sample(class_indices, n_samples)

    def _random_sample(self, X, target_size):
        return self.rng.choice(len(X), size=target_size, replace=False)

    def _is_classification(self, y):
        if y is None:
            return False
        unique_values = len(np.unique(y))
        n_samples = len(y)
        return unique_values < max(20, 0.05 * n_samples) or np.all(
            y == y.astype(int)
        )

    def _extract(self, data, indices):
        if data is None:
            return None
        return data.iloc[indices] if hasattr(data, "iloc") else data[indices]

    def _create_metadata(
        self,
        original_size,
        final_size,
        y_original=None,
        strategy="",
        y_sampled=None,
    ):
        def get_distribution(y):
            if y is None:
                return None
            unique, counts = np.unique(y, return_counts=True)
            return dict(zip(unique, counts))

        return {
            "sampled": final_size < original_size,
            "original_size": original_size,
            "final_size": final_size,
            "sampling_ratio": final_size / original_size,
            "strategy": strategy,
            "original_distribution": get_distribution(y_original),
            "final_distribution": get_distribution(y_sampled),
        }


def sample(X, y=None, **kwargs):
    """
    Sampling that preserves important patterns.

    Args:
        X: Feature matrix
        y: Target vector (optional)
        **kwargs: Sampler options
            - target_size: Explicit target size
            - sampling_ratio: Fraction to keep (default: 0.5)
            - min_samples_per_class: Min per class (default: 500)
            - random_state: Random seed (default: 42)

    Returns:
        tuple: (X_sampled, y_sampled, metadata)

    Examples:
        >>> # Keep 50% of data (default)
        >>> X_s, y_s, meta = sample(X, y)
        >>>
        >>> # Keep 30% of data
        >>> X_s, y_s, meta = sample(X, y, sampling_ratio=0.3)
        >>>
        >>> # Keep exactly 10000 samples
        >>> X_s, y_s, meta = sample(X, y, target_size=10000)
    """
    sampler = Sampler(**kwargs)
    return sampler.sample(X, y)
