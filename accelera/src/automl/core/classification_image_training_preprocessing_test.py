import shutil
import tempfile

import numpy as np
import pytest
from PIL import Image

from accelera.src.automl.core.classification_image_training_preprocessing import (  # noqa: E501
    ClassificationImageTrainingPreprocessing,
)
from accelera.src.automl.utils.preprocessing import check_path_exists


class TestClassificationImageTrainingPreprocessing:
    @pytest.fixture(autouse=True)
    def temp_folder(self):
        self.temp_dir = tempfile.mkdtemp()
        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    def test_constractor(self, tmp_path):
        training_folder = tmp_path / "training_folder"
        training_folder.mkdir()

        validation_folder = tmp_path / "validation_folder"
        validation_folder.mkdir()
        with pytest.raises(
            ValueError,
            match="Training Folder dosen't have any folder inside it ",
        ):
            ClassificationImageTrainingPreprocessing(
                training_folder_images=training_folder,
                folder_path=self.temp_dir,
            )

        (training_folder / "cats").mkdir()
        (training_folder / "dogs").mkdir()
        with pytest.raises(
            ValueError,
            match="Validation Folder dosen't have any folder inside it ",
        ):
            ClassificationImageTrainingPreprocessing(
                training_folder_images=training_folder,
                validation_folder_images=validation_folder,
                folder_path=self.temp_dir,
            )
        (validation_folder / "cats").mkdir()
        (validation_folder / "dogs").mkdir()
        (validation_folder / "birds").mkdir()
        with pytest.raises(
            ValueError, match=r"This category birds not in the training .*"
        ):
            ClassificationImageTrainingPreprocessing(
                training_folder_images=training_folder,
                validation_folder_images=validation_folder,
                folder_path=self.temp_dir,
            )

    def test_get_classes_mapping(self, tmp_path):
        training_folder = tmp_path / "training_folder"
        training_folder.mkdir()
        (training_folder / "cats").mkdir()
        (training_folder / "dogs").mkdir()
        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=training_folder, folder_path=self.temp_dir
        )
        preprocessor.get_classes_mapping()
        assert len(preprocessor.class2label_mapping) == 2
        assert len(preprocessor.label2class_mapping) == 2
        assert "cats" in preprocessor.class2label_mapping
        assert "dogs" in preprocessor.class2label_mapping
        assert preprocessor.class2label_mapping["cats"] == 0
        assert preprocessor.class2label_mapping["dogs"] == 1
        assert preprocessor.label2class_mapping[0] == "cats"
        assert preprocessor.label2class_mapping[1] == "dogs"
        assert check_path_exists(self.temp_dir, "class2label_mapping.pkl")
        assert check_path_exists(self.temp_dir, "label2class_mapping.pkl")

    def test_data_preparing(self, tmp_path):
        training_folder = tmp_path / "training_folder"
        training_folder.mkdir()
        cats_folder = training_folder / "cats"
        cats_folder.mkdir()
        dogs_folder = training_folder / "dogs"
        dogs_folder.mkdir()
        for i in range(5):
            (cats_folder / f"invalid_cat_{i}.png").touch()
            (dogs_folder / f"invalid_dog_{i}.png").touch()
        with pytest.raises(ValueError, match="There is no valid path"):
            ClassificationImageTrainingPreprocessing(
                training_folder_images=training_folder,
                folder_path=self.temp_dir,
            ).common_preprocessing()

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(cats_folder / f"cat_{i}.png")
            img.save(dogs_folder / f"dog_{i}.png")

        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=training_folder, folder_path=self.temp_dir
        )
        preprocessor.get_classes_mapping()
        invalid_images, invalid_labels = [], []
        images, labels = preprocessor.data_preparing(
            training_folder,
            invalid_images,
            invalid_labels,
            preprocessor.training_class,
        )
        assert "cats" in preprocessor.training_class
        assert "dogs" in preprocessor.training_class
        assert len(invalid_labels) == 10
        assert len(invalid_images) == 10
        assert len(images) == 10
        assert len(labels) == 10

    def test_data_overview(self, tmp_path):
        training_folder = tmp_path / "training_folder"
        training_folder.mkdir()
        cats_folder = training_folder / "cats"
        cats_folder.mkdir()
        dogs_folder = training_folder / "dogs"
        dogs_folder.mkdir()
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(cats_folder / f"cat_{i}.png")
            img.save(dogs_folder / f"dog_{i}.png")

        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=training_folder, folder_path=self.temp_dir
        )
        preprocessor.common_preprocessing()
        assert "training_folder" in preprocessor.report_data["data_overview"]
        assert (
            "validation_folder" not in preprocessor.report_data["data_overview"]
        )
        assert (
            preprocessor.report_data["data_overview"]["training_folder"][
                "images_len"
            ]
            == 10
        )
        assert (
            preprocessor.report_data["data_overview"]["training_folder"][
                "invalid_len"
            ]
            == 0
        )

    def test_splitting(self, tmp_path):
        training_folder = tmp_path / "training_folder"
        training_folder.mkdir()
        cats_folder = training_folder / "cats"
        cats_folder.mkdir()
        dogs_folder = training_folder / "dogs"
        dogs_folder.mkdir()
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(cats_folder / f"cat_{i}.png")
            img.save(dogs_folder / f"dog_{i}.png")

        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=training_folder, folder_path=self.temp_dir
        )
        preprocessor.common_preprocessing()
        assert len(preprocessor.training_paths) == 10
        assert len(preprocessor.training_labels) == 10
        assert preprocessor.validation_paths is None
        assert preprocessor.validation_labels is None
        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=training_folder,
            folder_path=self.temp_dir,
            split_training=True,
        )
        preprocessor.common_preprocessing()
        assert len(preprocessor.training_paths) == 8
        assert len(preprocessor.training_labels) == 8
        assert len(preprocessor.validation_paths) == 2
        assert len(preprocessor.validation_labels) == 2

    def test_get_loaders(self, tmp_path):
        training_folder = tmp_path / "training_folder"
        training_folder.mkdir()
        cats_folder = training_folder / "cats"
        cats_folder.mkdir()
        dogs_folder = training_folder / "dogs"
        dogs_folder.mkdir()
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(cats_folder / f"cat_{i}.png")
            img.save(dogs_folder / f"dog_{i}.png")

        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=training_folder, folder_path=self.temp_dir
        )
        preprocessor.common_preprocessing()
        assert preprocessor.training_loader is not None
        assert preprocessor.validation_loader is None
        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=training_folder,
            folder_path=self.temp_dir,
            split_training=True,
        )
        preprocessor.common_preprocessing()
        assert preprocessor.training_loader is not None
        assert preprocessor.validation_loader is not None
