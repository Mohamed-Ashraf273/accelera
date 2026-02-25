import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch
from accelera.src.automl.core.image_dataset import (
    ImageDataset,
)

class TestImageDataset:
    @pytest.fixture(autouse=True)
    def create_dummy_image(self):
        self.img = np.random.randint(0, 256, (125, 125, 3), dtype=np.uint8)
        self.img = Image.fromarray(self.img)

    def test_length(self):
        paths = ["imag1", "imag2", "imag3"]
        dataset = ImageDataset(image_paths=paths)
        assert len(dataset) == 3

    def test_random_brightness(self):
        dataset = ImageDataset(image_paths=[], brightness=False)
        result_img = dataset.random_brightness(self.img)
        assert np.array_equal(np.array(self.img), np.array(result_img))
        dataset = ImageDataset(
            image_paths=[], brightness=True, augmentation_probability=0
        )
        result_img = dataset.random_brightness(self.img)
        assert np.array_equal(np.array(self.img), np.array(result_img))
        dataset = ImageDataset(
            image_paths=[],
            brightness=True,
            augmentation_probability=1,
            brightness_factors=(0.5, 0.5),
        )
        with patch("random.random", return_value=0.0):
            result_img = dataset.random_brightness(self.img)
        assert not np.array_equal(np.array(self.img), np.array(result_img))

    def test_random_contrast(self):
        dataset = ImageDataset(image_paths=[], contrast=False)
        result_img = dataset.random_contrast(self.img)
        assert np.array_equal(np.array(self.img), np.array(result_img))
        dataset = ImageDataset(
            image_paths=[], contrast=True, augmentation_probability=0
        )
        result_img = dataset.random_contrast(self.img)
        assert np.array_equal(np.array(self.img), np.array(result_img))
        dataset = ImageDataset(
            image_paths=[],
            contrast=True,
            augmentation_probability=1,
            contrast_factors=(0.5, 0.5),
        )
        with patch("random.random", return_value=0.0):
            result_img = dataset.random_contrast(self.img)
        assert not np.array_equal(np.array(self.img), np.array(result_img))
