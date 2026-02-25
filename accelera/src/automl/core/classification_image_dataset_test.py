import pytest
import numpy as np
from PIL import Image
import torch
from unittest.mock import patch

from accelera.src.automl.core.classification_image_dataset import (
    ClassificationImageDataset,
)


class TestClassificationImageDataset:
    @pytest.fixture(autouse=True)
    def create_sample_data(self, tmp_path):
        self.paths = []
        self.labels = []
        for i in range(5):
            path = tmp_path / f"img{i}.png"
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img)
            img.save(path)
            self.paths.append(str(path))
            self.labels.append(i % 2)

    def test___getitem__(self):
        dataset = ClassificationImageDataset(
            image_paths=self.paths,
            labels=self.labels,
            augment=False,
            image_size=(125, 125),
        )
        img_tensor, label_tensor = dataset[0]
        assert len(dataset) == len(self.paths)
        assert img_tensor.shape[0] == 3
        assert img_tensor.shape[1] == 125
        assert img_tensor.shape[2] == 125
        assert label_tensor.item() == self.labels[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert isinstance(label_tensor, torch.Tensor)

    def test_random_horizontal_flip(self):
        img = Image.open(self.paths[0]).convert("RGB")
        dataset = ClassificationImageDataset(
            [], [], horizontal_flip=False, augmentation_probability=1
        )
        returned_img = dataset.random_horizontal_flip(img)
        assert np.array_equal(np.array(img), np.array(returned_img))
        dataset = ClassificationImageDataset(
            [], [], horizontal_flip=True, augmentation_probability=0
        )
        returned_img = dataset.random_horizontal_flip(img)
        assert np.array_equal(np.array(img), np.array(returned_img))
        dataset = ClassificationImageDataset(
            [], [], horizontal_flip=True, augmentation_probability=1
        )
        with patch("random.random", return_value=0.0):
            returned_img = dataset.random_horizontal_flip(img)
            returned_img2 = dataset.random_horizontal_flip(returned_img)
        assert np.array_equal(np.array(img), np.array(returned_img2))

    def test_random_vertical_flip(self):
        img = Image.open(self.paths[0]).convert("RGB")
        dataset = ClassificationImageDataset(
            [], [], vertical_flip=False, augmentation_probability=1
        )
        returned_img = dataset.random_vertical_flip(img)
        assert np.array_equal(np.array(img), np.array(returned_img))
        dataset = ClassificationImageDataset(
            [], [], vertical_flip=True, augmentation_probability=0
        )
        returned_img = dataset.random_vertical_flip(img)
        assert np.array_equal(np.array(img), np.array(returned_img))
        dataset = ClassificationImageDataset(
            [], [], vertical_flip=True, augmentation_probability=1
        )
        with patch("random.random", return_value=0.0):
            returned_img = dataset.random_vertical_flip(img)
            returned_img2 = dataset.random_vertical_flip(returned_img)
        assert np.array_equal(np.array(img), np.array(returned_img2))

    def test_random_rotation(self):
        img = Image.open(self.paths[0]).convert("RGB")
        dataset = ClassificationImageDataset(
            [], [], rotation=False, augmentation_probability=1
        )
        returned_img = dataset.random_rotation(img)
        assert np.array_equal(np.array(img), np.array(returned_img))
        dataset = ClassificationImageDataset(
            [], [], rotation=True, augmentation_probability=0
        )
        returned_img = dataset.random_rotation(img)
        assert np.array_equal(np.array(img), np.array(returned_img))
        dataset = ClassificationImageDataset(
            [], [], rotation=True, augmentation_probability=1
        )
        with (
            patch("random.random", return_value=0.0),
            patch("random.uniform", return_value=30.0),
        ):
            returned_img = dataset.random_rotation(img)
        assert not np.array_equal(np.array(img), np.array(returned_img))
