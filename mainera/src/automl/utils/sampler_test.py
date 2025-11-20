import numpy as np
import pytest
from sklearn.datasets import make_classification

from mainera.src.automl.utils.sampler import Sampler
from mainera.src.automl.utils.sampler import sample


class TestSampler:
    @pytest.fixture
    def sample_data(self):
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=3,
            n_informative=15,
            random_state=42,
        )
        return X, y

    def test_sampler_initialization(self):
        sampler = Sampler()
        assert sampler.sampling_ratio == 0.5
        assert sampler.min_samples_per_class == 50
        assert sampler.random_state == 42

    def test_sampling_reduces_size(self, sample_data):
        X, y = sample_data
        sampler = Sampler(sampling_ratio=0.3, random_state=42)
        X_sampled, y_sampled, meta = sampler.sample(X, y)

        assert len(X_sampled) < len(X)
        assert len(X_sampled) == len(y_sampled)
        assert meta["sampled"] is True

    def test_preserves_class_distribution(self, sample_data):
        X, y = sample_data
        sampler = Sampler(sampling_ratio=0.4, random_state=42)
        _, y_sampled, _ = sampler.sample(X, y)

        unique_orig, counts_orig = np.unique(y, return_counts=True)
        ratios_orig = counts_orig / len(y)

        unique_samp, counts_samp = np.unique(y_sampled, return_counts=True)
        ratios_samp = counts_samp / len(y_sampled)

        for i in range(len(ratios_orig)):
            assert abs(ratios_orig[i] - ratios_samp[i]) < 0.05

    def test_metadata_fields(self, sample_data):
        X, y = sample_data
        sampler = Sampler(sampling_ratio=0.5, random_state=42)
        _, _, meta = sampler.sample(X, y)

        assert "sampled" in meta
        assert "original_size" in meta
        assert "final_size" in meta
        assert "sampling_ratio" in meta
        assert "strategy" in meta
        assert meta["original_size"] == 1000

    def test_reproducibility(self, sample_data):
        X, y = sample_data

        sampler1 = Sampler(sampling_ratio=0.3, random_state=42)
        X_samp1, _, _ = sampler1.sample(X, y)

        sampler2 = Sampler(sampling_ratio=0.3, random_state=42)
        X_samp2, _, _ = sampler2.sample(X, y)

        np.testing.assert_array_equal(X_samp1, X_samp2)

    def test_target_size_parameter(self, sample_data):
        X, y = sample_data
        target = 300
        sampler = Sampler(target_size=target, random_state=42)
        X_sampled, _, meta = sampler.sample(X, y)

        assert abs(len(X_sampled) - target) < target * 0.15
        assert meta["final_size"] == len(X_sampled)


class TestSampleFunction:
    def test_sample_function_basic(self):
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=3,
            n_informative=15,
            random_state=42,
        )
        X_sampled, y_sampled, meta = sample(X, y, sampling_ratio=0.3)

        assert len(X_sampled) < len(X)
        assert len(X_sampled) == len(y_sampled)
        assert meta["sampled"] is True

    def test_sample_without_target(self):
        X = np.random.randn(1000, 10)
        X_sampled, y_sampled, meta = sample(X, y=None, sampling_ratio=0.3)

        assert len(X_sampled) == int(1000 * 0.3)
        assert y_sampled is None
        assert meta["strategy"] == "random_sampling"
