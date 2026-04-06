import shutil
import tempfile

import numpy as np
import pytest
from PIL import Image

from accelera.src.automl.core.segmentation_image_training_preprocessing import (
    SegmentationImageTrainingPreprocessing,
)
from accelera.src.utils.preprocessing import check_path_exists


class TestSegmentationImageTrainingPreprocessing:
    @pytest.fixture(autouse=True)
    def temp_folder(self):
        self.temp_dir = tempfile.mkdtemp()
        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    def test_constractor(self, tmp_path):
        training_folder_images = tmp_path / "training_folder_images"
        training_folder_images.mkdir()
        training_folder_masks = tmp_path / "training_folder_masks"
        training_folder_masks.mkdir()

        validation_folder_images = tmp_path / "validation_folder_images"
        validation_folder_images.mkdir()
        validation_folder_masks = tmp_path / "validation_folder_masks"
        validation_folder_masks.mkdir()
        with pytest.raises(
            ValueError, match="Training folder masks must be not null"
        ):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=None,
                folder_path=self.temp_dir,
            )

        with pytest.raises(
            ValueError,
            match="training folder images and training folder masks must be "
            "different",
        ):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=training_folder_images,
                folder_path=self.temp_dir,
            )

        with pytest.raises(ValueError, match=r".*does not exist"):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks="invalid",
                folder_path=self.temp_dir,
            )

        with pytest.raises(
            ValueError, match=r"you must add binary_mask_threshold.*"
        ):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=training_folder_masks,
                folder_path=self.temp_dir,
                binary_mask_threshold=None,
            )
        with pytest.raises(
            ValueError,
            match=r"binary_mask_threshold must be integer between 0 and 255",
        ):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=training_folder_masks,
                folder_path=self.temp_dir,
                binary_mask_threshold="invalid",
            )
        with pytest.raises(
            ValueError,
            match=r"binary_mask_threshold must be integer between 0 and 255",
        ):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=training_folder_masks,
                folder_path=self.temp_dir,
                binary_mask_threshold=-1,
            )
        with pytest.raises(
            ValueError, match=r"Validation folder masks must be not null"
        ):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=training_folder_masks,
                validation_folder_images=validation_folder_images,
                validation_folder_masks=None,
                folder_path=self.temp_dir,
                binary_mask_threshold=0,
            )
            with pytest.raises(ValueError, match=r".*does not exist"):
                SegmentationImageTrainingPreprocessing(
                    training_folder_images=training_folder_images,
                    training_folder_masks=training_folder_masks,
                    validation_folder_images=validation_folder_images,
                    validation_folder_masks="invalid",
                    folder_path=self.temp_dir,
                )
            with pytest.raises(
                ValueError,
                match="training folder images and training folder masks "
                "must be different",
            ):
                SegmentationImageTrainingPreprocessing(
                    training_folder_images=training_folder_images,
                    training_folder_masks=training_folder_images,
                    validation_folder_images=validation_folder_images,
                    validation_folder_masks=validation_folder_images,
                    folder_path=self.temp_dir,
                )
            assert check_path_exists(self.temp_dir, "data_info.pkl")

    def test_data_preparing(self, tmp_path):
        training_folder_images = tmp_path / "training_folder_images"
        training_folder_images.mkdir()
        training_folder_masks = tmp_path / "training_folder_masks"
        training_folder_masks.mkdir()

        validation_folder_images = tmp_path / "validation_folder_images"
        validation_folder_images.mkdir()
        validation_folder_masks = tmp_path / "validation_folder_masks"
        validation_folder_masks.mkdir()
        for i in range(5):
            (training_folder_images / f"invalid{i}.png").touch()
            (training_folder_masks / f"invalid_mask{i}.png").touch()

        with pytest.raises(
            ValueError, match="no matches between masks and images names"
        ):
            preprocessor = SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=training_folder_masks,
                folder_path=self.temp_dir,
            )
            preprocessor.common_preprocessing()
        for i in range(5):
            (training_folder_images / f"invalid{i}.png").touch()
            (training_folder_masks / f"invalid{i}.png").touch()
        with pytest.raises(ValueError, match="There is no valid path"):
            SegmentationImageTrainingPreprocessing(
                training_folder_images=training_folder_images,
                training_folder_masks=training_folder_masks,
                folder_path=self.temp_dir,
            ).common_preprocessing()

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            mask = Image.fromarray(
                np.random.randint(0, 256, (225, 225), dtype=np.uint8)
            )
            img.save(training_folder_images / f"img_{i}.png")
            mask.save(training_folder_masks / f"img_{i}.png")
        preprocessor = SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            folder_path=self.temp_dir,
        )
        invalid_images, invalid_masks = [], []
        training_images, training_masks = preprocessor.data_preparing(
            images_folder_path=training_folder_images,
            masks_folder_path=training_folder_masks,
            invalid_masks_paths=invalid_masks,
            invalid_images_paths=invalid_images,
        )
        assert len(invalid_images) == 5
        assert len(invalid_masks) == 5
        assert len(training_images) == 5
        assert len(training_masks) == 5

    def test_data_overview(self, tmp_path):
        training_folder_images = tmp_path / "training_folder_images"
        training_folder_images.mkdir()
        training_folder_masks = tmp_path / "training_folder_masks"
        training_folder_masks.mkdir()

        validation_folder_images = tmp_path / "validation_folder_images"
        validation_folder_images.mkdir()
        validation_folder_masks = tmp_path / "validation_folder_masks"
        validation_folder_masks.mkdir()
        for i in range(5):
            (training_folder_images / f"invalid{i}.png").touch()
            (training_folder_masks / f"invalid{i}.png").touch()
            (validation_folder_images / f"invalid{i}.png").touch()
            (validation_folder_masks / f"invalid{i}.png").touch()

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            mask = Image.fromarray(
                np.random.randint(0, 256, (225, 225), dtype=np.uint8)
            )
            img.save(training_folder_images / f"img_{i}.png")
            mask.save(training_folder_masks / f"img_{i}.png")
            img.save(validation_folder_images / f"img_{i}.png")
            mask.save(validation_folder_masks / f"img_{i}.png")

        preprocessor = SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            validation_folder_images=validation_folder_images,
            validation_folder_masks=validation_folder_masks,
            folder_path=self.temp_dir,
        )
        preprocessor.common_preprocessing()
        assert "training_folder" in preprocessor.report_data["data_overview"]
        assert "validation_folder" in preprocessor.report_data["data_overview"]
        assert (
            preprocessor.report_data["data_overview"]["training_folder"][
                "images_len"
            ]
            == 5
        )
        assert (
            preprocessor.report_data["data_overview"]["training_folder"][
                "invalid_len"
            ]
            == 5
        )
        assert (
            preprocessor.report_data["data_overview"]["validation_folder"][
                "images_len"
            ]
            == 5
        )
        assert (
            preprocessor.report_data["data_overview"]["validation_folder"][
                "invalid_len"
            ]
            == 5
        )

    def test_splitting(self, tmp_path):
        training_folder_images = tmp_path / "training_folder_images"
        training_folder_images.mkdir()
        training_folder_masks = tmp_path / "training_folder_masks"
        training_folder_masks.mkdir()

        validation_folder_images = tmp_path / "validation_folder_images"
        validation_folder_images.mkdir()
        validation_folder_masks = tmp_path / "validation_folder_masks"
        validation_folder_masks.mkdir()
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            mask = Image.fromarray(
                np.random.randint(0, 256, (225, 225), dtype=np.uint8)
            )
            img.save(training_folder_images / f"img_{i}.png")
            mask.save(training_folder_masks / f"img_{i}.png")
            img.save(validation_folder_images / f"img_{i}.png")
            mask.save(validation_folder_masks / f"img_{i}.png")

        preprocessor = SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            folder_path=self.temp_dir,
        )
        preprocessor.common_preprocessing()
        assert len(preprocessor.training_paths) == 10
        assert len(preprocessor.training_labels) == 10
        assert preprocessor.validation_paths is None
        assert preprocessor.validation_labels is None
        preprocessor = SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            folder_path=self.temp_dir,
            split_training=True,
        )
        preprocessor.common_preprocessing()
        assert len(preprocessor.training_paths) == 8
        assert len(preprocessor.training_labels) == 8
        assert len(preprocessor.validation_paths) == 2
        assert len(preprocessor.validation_labels) == 2

    def test_get_loaders(self, tmp_path):
        training_folder_images = tmp_path / "training_folder_images"
        training_folder_images.mkdir()
        training_folder_masks = tmp_path / "training_folder_masks"
        training_folder_masks.mkdir()

        validation_folder_images = tmp_path / "validation_folder_images"
        validation_folder_images.mkdir()
        validation_folder_masks = tmp_path / "validation_folder_masks"
        validation_folder_masks.mkdir()
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            mask = Image.fromarray(
                np.random.randint(0, 256, (225, 225), dtype=np.uint8)
            )
            img.save(training_folder_images / f"img_{i}.png")
            mask.save(training_folder_masks / f"img_{i}.png")
            img.save(validation_folder_images / f"img_{i}.png")
            mask.save(validation_folder_masks / f"img_{i}.png")

        preprocessor = SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            folder_path=self.temp_dir,
        )
        preprocessor.common_preprocessing()
        assert preprocessor.training_loader is not None
        assert preprocessor.validation_loader is None
        preprocessor = SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            folder_path=self.temp_dir,
            split_training=True,
        )
        preprocessor.common_preprocessing()
        assert preprocessor.training_loader is not None
        assert preprocessor.validation_loader is not None
        preprocessor = SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            validation_folder_images=validation_folder_images,
            validation_folder_masks=validation_folder_masks,
            folder_path=self.temp_dir,
        )
        preprocessor.common_preprocessing()
        assert preprocessor.training_loader is not None
        assert preprocessor.validation_loader is not None
