from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from accelera.src.automl.core.segmentation_image_dataset import (
    SegmentationImageDataset,
)


class TestSegmentationImageDataset:
    @pytest.fixture(autouse=True)
    def create_sample_data(self, tmp_path):
        self.paths = []
        self.mask_paths = []
        for i in range(5):
            path = tmp_path / f"img{i}.png"
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img)
            img.save(path)
            self.paths.append(str(path))
            mask_path = tmp_path / f"mask{i}.png"
            mask = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            mask = Image.fromarray(mask)
            mask.save(mask_path)
            self.mask_paths.append(str(mask_path))

    def test___getitem__(self):
        dataset = SegmentationImageDataset(
            image_paths=self.paths,
            masks=self.mask_paths,
            augment=False,
            image_size=(125, 125),
        )
        img_tensor, mask_tensor = dataset[0]
        assert len(dataset) == len(self.paths)
        assert img_tensor.shape[0] == 3
        assert img_tensor.shape[1] == 125
        assert img_tensor.shape[2] == 125
        assert mask_tensor.shape[0] == 1
        assert mask_tensor.shape[1] == 125
        assert mask_tensor.shape[2] == 125
        assert torch.all((mask_tensor == 1) | (mask_tensor == 0))
        assert isinstance(img_tensor, torch.Tensor)
        assert isinstance(mask_tensor, torch.Tensor)
        dataset = SegmentationImageDataset(
            image_paths=self.paths,
            masks=None,
            augment=False,
            image_size=(125, 125),
        )
        img_tensor, mask_tensor = dataset[0]
        assert len(dataset) == len(self.paths)
        assert img_tensor.shape[0] == 3
        assert img_tensor.shape[1] == 125
        assert img_tensor.shape[2] == 125
        assert mask_tensor is None
        assert isinstance(img_tensor, torch.Tensor)

    def test_random_horizontal_flip(self):
        img = Image.open(self.paths[0]).convert("RGB")
        mask = Image.open(self.mask_paths[0]).convert("L")
        dataset = SegmentationImageDataset(
            [], [], horizontal_flip=False, augmentation_probability=1
        )
        returned_img, returned_mask = dataset.random_horizontal_flip(img, mask)
        assert np.array_equal(np.array(img), np.array(returned_img))
        assert np.array_equal(np.array(mask), np.array(returned_mask))
        dataset = SegmentationImageDataset(
            [], [], horizontal_flip=True, augmentation_probability=0
        )
        returned_img, returned_mask = dataset.random_horizontal_flip(img, mask)
        assert np.array_equal(np.array(img), np.array(returned_img))
        assert np.array_equal(np.array(mask), np.array(returned_mask))
        dataset = SegmentationImageDataset(
            [], [], horizontal_flip=True, augmentation_probability=1
        )
        with patch("random.random", return_value=0.0):
            returned_img, returned_mask = dataset.random_horizontal_flip(img, mask)
            returned_img2, returned_mask2 = dataset.random_horizontal_flip(
                returned_img, returned_mask
            )
        assert np.array_equal(np.array(img), np.array(returned_img2))
        assert np.array_equal(np.array(mask), np.array(returned_mask2))
        dataset = SegmentationImageDataset(
            [], [], horizontal_flip=True, augmentation_probability=1
        )
        returned_img, returned_mask = dataset.random_horizontal_flip(img, None)
        assert not np.array_equal(np.array(img), np.array(returned_img))
        assert returned_mask is None

    def test_random_vertical_flip(self):
        img = Image.open(self.paths[0]).convert("RGB")
        mask = Image.open(self.mask_paths[0]).convert("L")

        dataset = SegmentationImageDataset(
            [], [], vertical_flip=False, augmentation_probability=1
        )
        returned_img, returned_mask = dataset.random_vertical_flip(img, mask)
        assert np.array_equal(np.array(img), np.array(returned_img))
        assert np.array_equal(np.array(mask), np.array(returned_mask))

        dataset = SegmentationImageDataset(
            [], [], vertical_flip=True, augmentation_probability=0
        )
        returned_img, returned_mask = dataset.random_vertical_flip(img, mask)
        assert np.array_equal(np.array(img), np.array(returned_img))
        assert np.array_equal(np.array(mask), np.array(returned_mask))
        dataset = SegmentationImageDataset(
            [], [], vertical_flip=True, augmentation_probability=1
        )
        with patch("random.random", return_value=0.0):
            returned_img, returned_mask = dataset.random_vertical_flip(img, mask)
            returned_img2, returned_mask2 = dataset.random_vertical_flip(
                returned_img, returned_mask
            )
        assert np.array_equal(np.array(img), np.array(returned_img2))
        assert np.array_equal(np.array(mask), np.array(returned_mask2))
        dataset = SegmentationImageDataset(
            [], [], vertical_flip=True, augmentation_probability=1
        )
        returned_img, returned_mask = dataset.random_vertical_flip(img, None)
        assert not np.array_equal(np.array(img), np.array(returned_img))
        assert returned_mask is None

    def test_random_rotation(self):
        img = Image.open(self.paths[0]).convert("RGB")
        mask = Image.open(self.mask_paths[0]).convert("L")

        dataset = SegmentationImageDataset(
            [], [], rotation=False, augmentation_probability=1
        )
        returned_img, returned_mask = dataset.random_rotation(img, mask)
        assert np.array_equal(np.array(img), np.array(returned_img))
        assert np.array_equal(np.array(mask), np.array(returned_mask))
        dataset = SegmentationImageDataset(
            [], [], rotation=True, augmentation_probability=0
        )
        returned_img, returned_mask = dataset.random_rotation(img, mask)
        assert np.array_equal(np.array(img), np.array(returned_img))
        assert np.array_equal(np.array(mask), np.array(returned_mask))
        dataset = SegmentationImageDataset(
            [], [], rotation=True, augmentation_probability=1
        )
        with (
            patch("random.random", return_value=0.0),
            patch("random.uniform", return_value=30.0),
        ):
            returned_img, returned_mask = dataset.random_rotation(img, mask)
        assert not np.array_equal(np.array(img), np.array(returned_img))
        assert not np.array_equal(np.array(mask), np.array(returned_mask))
        dataset = SegmentationImageDataset(
            [], [], rotation=True, augmentation_probability=1
        )
        returned_img, returned_mask = dataset.random_rotation(img, None)
        assert not np.array_equal(np.array(img), np.array(returned_img))
        assert returned_mask is None
