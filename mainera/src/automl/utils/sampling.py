import numpy as np

from mainera.src.utils.array_utils import convert_to_array


class Sampler:
    MIN_SAMPLES_PER_CLASS = 1000

    def __init__(
        self,
        target_size=None,
        min_samples_per_class=None,
        preserve_boundaries=True,
        preserve_diversity=True,
        random_state=42,
    ):
        """
        Initialize SmartSampler.

        Args:
            target_size: Target number of samples after sampling
            min_samples_per_class: Minimum samples per class for classification
            preserve_boundaries: Whether to preserve boundary samples
            preserve_diversity: Whether to preserve diverse samples
            random_state: Random state for reproducibility
        """
        self.target_size = target_size
        self.min_samples_per_class = (
            min_samples_per_class or self.MIN_SAMPLES_PER_CLASS
        )
        self.preserve_boundaries = preserve_boundaries
        self.preserve_diversity = preserve_diversity
        self.random_state = random_state

        self.is_sampled = False
        self.original_size = None
        self.original_distribution = None
        self.sampled_indices = None
        self.sampling_strategy = None

    def sample(self, X, y=None):
        """
        Perform smart sampling on the dataset.

        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target vector (numpy array or pandas Series), optional

        Returns:
            tuple: (X_sampled, y_sampled, metadata)
        """
        if self.target_size is None:
            self.target_size = len(X) // 2  # Default to 50% of original size

        X_array = convert_to_array(X)
        y_array = convert_to_array(y) if y is not None else None
        self.original_distribution = (
            self._get_distribution(y_array) if y_array is not None else None
        )

        metadata = {
            "sampled": False,
            "original_size": len(X_array),
            "original_distribution": self.original_distribution,
            "final_size": len(X_array),
            "final_distribution": self.original_distribution,
            "sampling_ratio": 1.0,
            "strategy": None,
        }

        self._adjust(X_array, y_array)

        if y_array is not None and self._is_classification(y_array):
            indices = self._classification_sample(X_array, y_array)
            self.sampling_strategy = "sampling_classification"
        else:
            indices = self._unsupervised_sample(X_array, y_array)
            self.sampling_strategy = "sampling_unsupervised"

        X_sampled = self._extract_samples(X, indices)
        y_sampled = self._extract_samples(y, indices) if y is not None else None

        self.is_sampled = True
        self.sampled_indices = indices

        metadata.update(
            {
                "sampled": True,
                "original_size": self.original_size,
                "original_distribution": self.original_distribution,
                "final_size": len(indices),
                "final_distribution": self._get_distribution(y_sampled)
                if y_sampled is not None
                else None,
                "sampling_ratio": len(indices) / self.original_size,
                "strategy": self.sampling_strategy,
            }
        )

        return X_sampled, y_sampled, metadata

    def _get_distribution(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def _adjust(self, X, y=None):
        n_samples = len(X)
        self.original_size = n_samples

        if y is not None and self._is_classification(y):
            unique, counts = np.unique(y, return_counts=True)
            min_class_count = counts.min()

            if min_class_count < self.min_samples_per_class:
                n_classes = len(unique)
                adjusted_target = max(
                    self.target_size, n_classes * self.min_samples_per_class
                )
                self.target_size = min(adjusted_target, n_samples)

    def _classification_sample(self, X, y):
        """
        Sampling for classification tasks.

        Strategy:
        1. Identify boundary samples (potentially hard to classify)
        2. Find representative samples (cluster centers within each class)
        3. Ensure class balance
        4. Add diverse samples to fill quota
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)

        samples_per_class = {}
        total_target = min(self.target_size, len(X))

        for cls, count in zip(unique_classes, class_counts):
            proportional = int((count / len(X)) * total_target)
            samples_per_class[cls] = max(
                proportional, self.min_samples_per_class
            )

        total_allocated = sum(samples_per_class.values())
        if total_allocated > total_target:
            scale = total_target / total_allocated
            samples_per_class = {
                cls: max(int(count * scale), self.min_samples_per_class)
                for cls, count in samples_per_class.items()
            }

        all_indices = []

        for cls in unique_classes:
            class_mask = y == cls
            class_indices = np.where(class_mask)[0]
            X_class = X[class_mask]

            n_samples_needed = min(samples_per_class[cls], len(class_indices))

            if len(class_indices) <= n_samples_needed:
                selected_indices = class_indices
            else:
                selected_indices = self._sample_from_class(
                    X_class, class_indices, n_samples_needed
                )

            all_indices.extend(selected_indices)

        return np.array(all_indices)

    def _sample_from_class(self, X_class, class_indices, n_samples):
        """
        Sampling within a single class.

        Strategy:
        1. Find cluster centers (representative samples)
        2. Find boundary samples (using density estimation)
        3. Add diverse samples
        """
        if len(X_class) <= n_samples:
            return class_indices

        selected_indices = []
        np.random.seed(self.random_state)

        n_centers = max(int(n_samples * 0.4), min(10, n_samples // 2))
        center_indices = self._find_representative_samples(X_class, n_centers)
        selected_indices.extend(class_indices[center_indices])

        if self.preserve_boundaries:
            n_boundary = int(n_samples * 0.3)
            remaining_mask = np.ones(len(X_class), dtype=bool)
            remaining_mask[center_indices] = False
            remaining_indices = np.where(remaining_mask)[0]

            if len(remaining_indices) > 0:
                boundary_indices = self._find_boundary_samples(
                    X_class[remaining_indices], n_boundary
                )
                selected_indices.extend(
                    class_indices[remaining_indices[boundary_indices]]
                )

        n_remaining = n_samples - len(selected_indices)
        if n_remaining > 0:
            already_selected = set(
                idx - class_indices[0] for idx in selected_indices
            )
            remaining = [
                i for i in range(len(X_class)) if i not in already_selected
            ]

            if len(remaining) > 0:
                if self.preserve_diversity and len(remaining) > n_remaining:
                    diverse_indices = self._find_diverse_samples(
                        X_class[remaining], n_remaining
                    )
                    selected_indices.extend(
                        class_indices[np.array(remaining)[diverse_indices]]
                    )
                else:
                    chosen = np.random.choice(
                        remaining,
                        size=min(n_remaining, len(remaining)),
                        replace=False,
                    )
                    selected_indices.extend(class_indices[chosen])

        return selected_indices[:n_samples]

    def _unsupervised_sample(self, X, y=None):
        """
        Smart sampling for unsupervised/regression tasks.

        Strategy:
        1. Find representative samples (cluster centers)
        2. Ensure diversity (maximum spread)
        3. Include some random samples for generalization
        """
        n_samples = min(self.target_size, len(X))

        n_centers = int(n_samples * 0.5)
        center_indices = self._find_representative_samples(X, n_centers)
        selected_indices = list(center_indices)

        remaining_mask = np.ones(len(X), dtype=bool)
        remaining_mask[center_indices] = False
        remaining_indices = np.where(remaining_mask)[0]

        n_remaining = n_samples - len(selected_indices)
        if len(remaining_indices) > 0 and n_remaining > 0:
            if self.preserve_diversity and len(remaining_indices) > n_remaining:
                diverse_indices = self._find_diverse_samples(
                    X[remaining_indices], n_remaining
                )
                selected_indices.extend(remaining_indices[diverse_indices])
            else:
                np.random.seed(self.random_state)
                chosen = np.random.choice(
                    remaining_indices,
                    size=min(n_remaining, len(remaining_indices)),
                    replace=False,
                )
                selected_indices.extend(chosen)

        return np.array(selected_indices[:n_samples])

    def _find_representative_samples(self, X, n_samples):
        if len(X) <= n_samples:
            return np.arange(len(X))

        try:
            np.random.seed(self.random_state)

            step = len(X) / n_samples
            base_indices = np.arange(0, len(X), step)[:n_samples].astype(int)

            jitter = np.random.randint(
                -int(step * 0.1), int(step * 0.1) + 1, size=len(base_indices)
            )
            indices = np.clip(base_indices + jitter, 0, len(X) - 1)

            indices = np.unique(indices)
            if len(indices) < n_samples:
                available = np.setdiff1d(np.arange(len(X)), indices)
                extra = np.random.choice(
                    available, size=n_samples - len(indices), replace=False
                )
                indices = np.concatenate([indices, extra])

            return indices[:n_samples]
        except Exception:
            np.random.seed(self.random_state)
            return np.random.choice(len(X), size=n_samples, replace=False)

    def _find_boundary_samples(self, X, n_samples):
        if len(X) <= n_samples:
            return np.arange(len(X))

        try:
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8

            normalized_diff = (X - mean) / std
            distances = np.linalg.norm(normalized_diff, axis=1)

            boundary_indices = np.argsort(distances)[-n_samples:]
            return boundary_indices
        except Exception:
            np.random.seed(self.random_state)
            return np.random.choice(len(X), size=n_samples, replace=False)

    def _find_diverse_samples(self, X, n_samples):
        if len(X) <= n_samples:
            return np.arange(len(X))

        try:
            np.random.seed(self.random_state)
            if X.shape[1] > 10:
                projection = X[:, : min(5, X.shape[1])].sum(axis=1)
            else:
                projection = X.sum(axis=1)

            sorted_indices = np.argsort(projection)

            step = len(sorted_indices) / n_samples
            selected_indices = [
                sorted_indices[int(i * step)] for i in range(n_samples)
            ]

            return np.array(selected_indices)
        except Exception:
            np.random.seed(self.random_state)
            return np.random.choice(len(X), size=n_samples, replace=False)

    def _is_classification(self, y):
        if y is None:
            return False

        y_array = np.asarray(y)
        unique_values = len(np.unique(y_array))
        n_samples = len(y_array)

        is_discrete = unique_values < max(20, 0.2 * n_samples)
        is_integer = np.all(y_array == y_array.astype(int))

        return is_discrete or is_integer

    def _extract_samples(self, data, indices):
        if data is None:
            return None
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]


def sample(X, y=None, task=None, **kwargs):
    """
    Convenience function for smart sampling.

    Args:
        X: Feature matrix
        y: Target vector (optional)
        task: Task type (optional, for logging)
        **kwargs: Additional arguments for SmartSampler

    Returns:
        tuple: (X_sampled, y_sampled, metadata)

    Example:
        >>> X_sampled, y_sampled, metadata = smart_sample(X, y)
        >>> print(f"Strategy: {metadata['strategy']}")
        >>> print(f"Kept {metadata['sampling_ratio']:.1%} of data")
    """
    sampler = Sampler(**kwargs)
    return sampler.sample(X, y)
