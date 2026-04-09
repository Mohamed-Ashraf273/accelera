import numpy as np
import pytest
from PIL import Image

from accelera.src.automl.core.segmentation_image_testing_preprocessing import (
    SegmentationImageTestingPreprocessing,
)
from accelera.src.utils.preprocessing import save_pickle


class TestSegmentationImageTestingPreprocessing:
    def test_constructor(self, tmp_path):
        data_info = {"image_size": (125, 125), "binary_mask_threshold": 128}
        save_pickle(tmp_path, data_info, "data_info.pkl")

        with pytest.raises(
            ValueError, match="Image paths must be list of paths not none"
        ):
            SegmentationImageTestingPreprocessing(
                image_paths=None, folder_path=tmp_path
            )

        with pytest.raises(ValueError, match="Image paths must be list of paths"):
            SegmentationImageTestingPreprocessing(
                image_paths="path", folder_path=tmp_path
            )
        with pytest.raises(ValueError, match="Image paths is empty list"):
            SegmentationImageTestingPreprocessing(
                image_paths=[], folder_path=tmp_path
            )

        with pytest.raises(ValueError, match="masks must be list of masks paths"):
            SegmentationImageTestingPreprocessing(
                image_paths=["path"], image_masks=0, folder_path=tmp_path
            )

        with pytest.raises(
            ValueError, match="image paths length must equal masks length"
        ):
            SegmentationImageTestingPreprocessing(
                image_paths=["path1", "path2"],
                image_masks=[0],
                folder_path=tmp_path,
            )
        with pytest.raises(ValueError, match="There is no valid image exists"):
            SegmentationImageTestingPreprocessing(
                image_paths=["path1", "path2"],
                image_masks=["mask", "mask"],
                folder_path=tmp_path,
            )
        images = []
        masks = []
        for i in range(5):
            (tmp_path / f"invalid_{i}.png").touch()
            images.append(str(tmp_path / f"invalid_{i}.png"))
            masks.append(str(tmp_path / f"invalid_{i}.png"))
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(tmp_path / f"img_{i}.png")
            images.append(str(tmp_path / f"img_{i}.png"))
            masks.append(str(tmp_path / f"img_{i}.png"))
        preprocesseor = SegmentationImageTestingPreprocessing(
            image_paths=images,
            image_masks=masks,
            folder_path=tmp_path,
        )
        assert len(preprocesseor.valid_images) == 5
        assert len(preprocesseor.invalid_images) == 5

    def test_common_preprocessing(self, tmp_path):
        data_info = {"image_size": (125, 125), "binary_mask_threshold": 128}
        save_pickle(tmp_path, data_info, "data_info.pkl")

        images = []
        masks = []
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(tmp_path / f"img_{i}.png")
            images.append(str(tmp_path / f"img_{i}.png"))
            masks.append(str(tmp_path / f"img_{i}.png"))

        testing_loader, invalid_paths = SegmentationImageTestingPreprocessing(
            image_paths=images,
            image_masks=None,
            folder_path=tmp_path,
        ).common_preprocessing()
        assert testing_loader is not None
        assert len(invalid_paths) == 0
        batch = next(iter(testing_loader))
        assert len(batch) == 4
        testing_loader, invalid_paths = SegmentationImageTestingPreprocessing(
            image_paths=images,
            image_masks=masks,
            folder_path=tmp_path,
        ).common_preprocessing()
        assert testing_loader is not None
        assert len(invalid_paths) == 0
        batch = next(iter(testing_loader))
        assert len(batch) == 2
        assert len(batch[0]) == 4
        assert len(batch[1]) == 4
