import pytest
import shutil
import tempfile
import pandas as pd
from accelera.src.automl.core.image_training_preprocessing import (
    ImageTrainingPreprocessing,
)


class TestImageTrainingPreprocessing:
    @pytest.fixture(autouse=True)
    def temp_folder(self):
        self.temp_dir = tempfile.mkdtemp()
        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    def test_image_training_preprocessing_initialization(self, tmp_path):
        training_folder = tmp_path / "training"
        training_folder.mkdir()
        validation_folder = tmp_path / "validation"
        validation_folder.mkdir()
        with pytest.raises(ValueError, match="Training folder must be not null"):
            ImageTrainingPreprocessing(
                training_folder=None,
                validation_folder=None,
                split_training=False,
                random_state=42,
                images_size=(224, 224),
                val_size=0.2,
                folder_path=self.temp_dir,
            )

        with pytest.raises(ValueError):
            ImageTrainingPreprocessing(
                training_folder="invalid",
                validation_folder=None,
                split_training=False,
                random_state=42,
                images_size=(224, 224),
                val_size=0.2,
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder="invalid",
                split_training=False,
                random_state=42,
                images_size=(224, 224),
                val_size=0.2,
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=True,
                random_state=42,
                images_size=(224, 224),
                val_size=0.2,
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=training_folder,
                split_training=False,
                random_state=42,
                images_size=(224, 224),
                val_size=0.2,
                folder_path=self.temp_dir,
            )
        with pytest.raises(
            ValueError, match="Test size is invalid it must be less than 0.5"
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size="invalid type",
                random_state=42,
                images_size=(224, 224),
                folder_path=self.temp_dir,
            )
        with pytest.raises(
            ValueError, match="Test size is invalid it must be less than 0.5"
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.9,
                random_state=42,
                images_size=(224, 224),
                folder_path=self.temp_dir,
            )
        with pytest.raises(
            ValueError, match="Random state is invalid it must be integer or None"
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42.4,
                images_size=(224, 224),
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError, match="Image size must be tuple"):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size="invalid",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError, match="Image size is not integer"):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(224, 224.3),
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError, match="Image size is not integer"):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(224.5, 224),
                folder_path=self.temp_dir,
            )
        with pytest.raises(
            ValueError,
            match=r".*than or equal 32 and less than or equal 1024",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(16, 224),
                folder_path=self.temp_dir,
            )
        with pytest.raises(
            ValueError,
            match="batch_size must be a integer",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size="invalid",
            )
        with pytest.raises(
            ValueError,
            match="batch_size must be a positive value",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=-1,
            )
        with pytest.raises(
            ValueError,
            match="augment must be a boolean",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment="invalid",
            )
        with pytest.raises(
            ValueError,
            match=r"horizontal_flip must be a boolean",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                horizontal_flip="invalid",
            )
        with pytest.raises(
            ValueError,
            match="vertical_flip must be a boolean",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                vertical_flip="invalid",
            )
        with pytest.raises(
            ValueError,
            match="rotation must be a boolean",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                rotation="invalid",
            )
        with pytest.raises(
            ValueError,
            match="brightness must be a boolean",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                brightness="invalid",
            )
        with pytest.raises(
            ValueError,
            match="contrast must be a boolean",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                contrast="invalid",
            )
        with pytest.raises(
            ValueError,
            match=r"augmentation_probability must be between \[0,1\]",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                augmentation_probability="invalid",
            )
        with pytest.raises(
            ValueError,
            match=r"augmentation_probability must be between \[0,1\]",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                augmentation_probability=2,
            )
        with pytest.raises(
            ValueError,
            match="rotation_angle must be positive integer or float",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                rotation_angle="invalid",
            )
        with pytest.raises(
            ValueError,
            match="rotation_angle must be positive integer or float",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                rotation_angle=-1,
            )
        with pytest.raises(
            ValueError,
            match="brightness_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                brightness_factors=1,
            )
        with pytest.raises(
            ValueError,
            match="brightness_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                brightness_factors=(2),
            )
        with pytest.raises(
            ValueError,
            match="brightness_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                brightness_factors=(2, 3, 4),
            )
        with pytest.raises(
            ValueError,
            match="brightness_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                brightness_factors=("invalid", 4),
            )

        with pytest.raises(
            ValueError,
            match="contrast_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                contrast_factors=1,
            )
        with pytest.raises(
            ValueError,
            match="contrast_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                contrast_factors=(2),
            )
        with pytest.raises(
            ValueError,
            match="contrast_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                contrast_factors=(2, 3, 4),
            )
        with pytest.raises(
            ValueError,
            match="contrast_factors must be tuple of two items float or integers",
        ):
            ImageTrainingPreprocessing(
                training_folder=training_folder,
                validation_folder=validation_folder,
                split_training=False,
                val_size=0.2,
                random_state=42,
                images_size=(32, 224),
                folder_path=self.temp_dir,
                batch_size=16,
                augment=True,
                contrast_factors=("invalid", 4),
            )

    def test_get_sample_random(self, tmp_path):
        training_folder = tmp_path / "training"
        training_folder.mkdir()
        training_preprocessor = ImageTrainingPreprocessing(
            training_folder=training_folder,
            validation_folder=None,
            folder_path=self.temp_dir,
            split_training=False,
            images_size=(224, 224),
            random_state=42,
            val_size=0.2,
        )
        paths = ["image1.png", "image2.png"]
        labels = [0, 1]
        df = training_preprocessor.get_sample_random(
            "training", paths, labels, num_samples=2
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["training_paths", "training_labels"]

    def test_splitting(self, tmp_path):
        training_folder = tmp_path / "training"
        training_folder.mkdir()
        training_preprocessor = ImageTrainingPreprocessing(
            training_folder=training_folder,
            validation_folder=None,
            folder_path=self.temp_dir,
            split_training=False,
            images_size=(224, 224),
            random_state=42,
            val_size=0.2,
        )
        training_folder_images_paths = [
            "imag1",
            "imag2",
            "imag3",
            "imag4",
            "imag5",
            "imag6",
            "imag7",
            "imag8",
            "imag9",
            "imag10",
        ]
        validation_folder_images_paths = ["imag11", "imag12"]
        training_folder_images_labels = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        validation_folder_images_labels = [1, 0]
        training_preprocessor.splitting(
            training_folder_images_paths,
            validation_folder_images_paths,
            training_folder_images_labels,
            validation_folder_images_labels,
        )
        assert len(training_preprocessor.training_paths) == 10
        assert training_preprocessor.training_paths == training_folder_images_paths
        assert len(training_preprocessor.training_labels) == 10
        assert training_preprocessor.training_labels == training_folder_images_labels
        assert len(training_preprocessor.validation_paths) == 2
        assert training_preprocessor.validation_paths == validation_folder_images_paths
        assert len(training_preprocessor.validation_labels) == 2
        assert (
            training_preprocessor.validation_labels == validation_folder_images_labels
        )
        training_preprocessor = ImageTrainingPreprocessing(
            training_folder=training_folder,
            validation_folder=None,
            folder_path=self.temp_dir,
            split_training=True,
            images_size=(224, 224),
            random_state=42,
            val_size=0.2,
        )
        training_preprocessor.splitting(
            training_folder_images_paths,
            validation_folder_images_paths,
            training_folder_images_labels,
            validation_folder_images_labels,
        )
        assert len(training_preprocessor.training_paths) == 8
        assert len(training_preprocessor.training_labels) == 8
        assert len(training_preprocessor.validation_paths) == 2
        assert len(training_preprocessor.validation_labels) == 2
